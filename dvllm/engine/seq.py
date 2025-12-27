from copy import copy
from enum import Enum, auto
from itertools import count

from dvllm.sparams import SParams


"""
Sequence 类的核心目的：
封装 prompt + 生成 token
管理 序列状态（等待 / 运行 / 完成）
管理 KV Cache 分块信息
支持 追加生成 token
支持 序列化 / 反序列化
换句话说：这个类就是 dvllm 内部 “每条生成请求的容器 + 缓存管理器”。
"""


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    """
    假设：
    self.token_ids = [0, 1, 2, ..., 499]  # 500 个 token
    self.block_size = 256
    self.num_blocks = 2
    block(0) → [0, 1, ..., 255] （第 1 个 block）
    block(1) → [256, 257, ..., 499] （最后一个 block，有 244 个 token）
    作用
    这个方法就是 把整个 token 序列按 block_size 切成小块，方便模型按 block 进行缓存、处理或并行计算。
    在生成模型里常用，比如 KV cache、attention window 都按 block 处理。
    """

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    """
    这是 往序列里“追加新 token” 的方法，同时更新序列状态，让 sequence 对象实时保持最新。
    """

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    """
    这里返回的元组包含：
    num_tokens：序列总长度
    num_prompt_tokens：提示 token 数量
    num_cached_tokens：已缓存 token 数
    block_table：block 缓存表
    token 数据：
    如果当前还没生成 completion token (num_completion_tokens == 0)，就存整个 token_ids 列表
    如果已经生成了 completion token，只存最后一个 token (last_token)
    优化点：一旦序列很长，只存最后一个 token 可以节省内存/存储空间，因为生成的 token 可能非常多，不需要全部序列都序列化。
    """

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    """
    功能：
    当你从序列化状态恢复对象时，Python 会调用 __setstate__
    state[:-1] 恢复前四个属性
    最后一个元素：
        如果没有生成 completion token，就把 token_ids 恢复成完整列表
        如果已经生成 completion token，只恢复 last_token`
    """

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

    """
    支持对象序列化/传输：可以用 pickle 存储或在多进程间传递 Sequence 对象
    节省内存：
        对于长序列，不必把所有生成 token 都存储，避免占用大量内存
    兼顾完整性和效率：
        对于还没生成 completion 的序列，必须完整保存 prompt token
        对于已经生成的序列，只关心最后 token，用于继续生成或统计
    让Sequence 对象可以高效地序列化和反序列化，同时对生成的长 token 序列做内存优化
    """
