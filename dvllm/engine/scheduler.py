from collections import deque
from dvllm.config import Config
from dvllm.engine.seq import Sequence, SequenceStatus
from dvllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    """
    这是一个 状态检查函数，通常用于主循环或批处理流程：
        - 如果 is_finished() 返回 True，没有等待的序列且没有正在运行的序列,生成引擎可以停止或者释放资源。也就是序列都已经生成完成。
    本质上就是：
        - “队列空了 + 没有人在跑 = 生成完成”
    """

    def is_finished(self):
        return not self.waiting and not self.running

    """
    新来的序列 → add → 排队等生成
    正在生成的序列被抢占 → preempt(preempt(seq) 会把序列从运行队列移到等待队列前端 (appendleft) → 插队回等待队列前面
    其实也就是把一个新序列或者被抢占的序列加入等待队列
    """

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    """
    排队 + batch 调度 + 资源管理的核心逻辑
    """

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 这个序列加进去会超过 batch 的 token 限制 或者 内存不够分配它 → 那就不要再往 batch 加序列了
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1  # 表示本次 batch 中已经确定要运行的序列数量增加 1。
            self.block_manager.allocate(
                seq
            )  # 这是 prefill 阶段，序列第一次进模型，需要完整计算注意力，所以必须分配完整新块。给这个序列分配 KV Cache（显存里的块）
            num_batched_tokens += (
                len(seq) - seq.num_cached_tokens
            )  # prefill：需要计算全部 token，所以加 len(seq)，decode：只计算 1 个新 token（cached tokens 用旧的）这里是 prefill，所以真正消耗的计算量是：序列长度 - 已缓存的 token 数
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        """
        只要还有正在运行的序列，并且本次 batch 调度的序列数量没满，就继续循环
            - self.running = 队里还有人
            - num_seqs < max_num_seqs = 你手里糖果有限，最多发这么多
        检查能否给它分配 block/缓存，如果不能：
            -把其它序列临时暂停（preempt）腾空间
            -如果没有其它序列，就先暂停自己
        如果可以分配，就正式安排它生成 token，并加入 scheduled 列表 
        """
        # decode
        while self.running and num_seqs < self.max_num_seqs:
            """
            从 self.running 队列的左边（最先进入的序列）取出一个序列 seq。
            popleft() 是双端队列（deque）的操作，表示 FIFO（先进先出）
            这个 while 循环的条件是：当前序列无法再向 block_manager 分配内存或缓存空间时
            can_append(seq) 检查能否为这个序列分配更多 block/内存。
            如果不能，就要采取措施，让其它序列腾空间。
            """
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    """
    preempt 的核心功能：
    把序列状态标记为 WAITING（暂停生成）。
    释放序列占用的缓存/内存资源。
    把序列放回等待队列，未来还会继续生成。
    用途：
        在batching 或并行生成 的场景下，如果当前 GPU/CPU 资源紧张，某些序列可以被“抢占”，让别的序列先运行。
        之后抢占的序列仍然可以继续生成，而不会浪费资源。
    """

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    """
    postprocess 的核心功能：
    将模型生成的 token 添加到对应的序列对象里。
    判断序列是否已经完成：
        生成了 EOS token（如果没有被忽略）
        或者达到了最大生成长度。
    如果完成：
        标记状态
        释放占用资源
        从“正在运行”的列表中移除
    本质：就是 管理生成序列的生命周期，保证生成过程高效、资源被及时回收。
    """

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
