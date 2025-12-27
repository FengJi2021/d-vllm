import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from dvllm.config import Config
from dvllm.sparams import SParams
from dvllm.engine.run_model import RunModel
from dvllm.engine.scheduler import Scheduler
from dvllm.engine.seq import Sequence


import logging

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, model, **kwargs):
        logger.info(f"model:{model}")
        logger.info(f"kwargs:{kwargs}")
        for field in fields(Config):
            logger.debug(f"field_name: {field.name}")
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        device_type = kwargs.get("device_type", "auto")
        config_kwargs["device_type"] = device_type
        logger.debug(
            f"fields={config_fields}, kwargs={config_kwargs},devices={device_type}"
        )
        config = Config(model, **config_kwargs)
        logger.debug(f"config={config}")
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        logger.debug(f"ctx:{ctx}")
        for i in range(1, config.tensor_parallel_size):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            logger.debug(f"process:{process}")
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = RunModel(config, 0, self.events)
        logging.debug(f"!!!!!!event:{self.events}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        logging.debug(f"scheduler:{self.scheduler}")
        # 支持诊断 dump_path
        self.dump_path = None
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.exit()
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sparams: SParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sparams)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill, self.dump_path)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sparams: SParams | list[SParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sparams, list):
            sparams = [sparams] * len(prompts)

        # Enqueue prompts as Sequence objects into the scheduler.
        # If we don't add the requests, the scheduler will be empty and
        # generate() will return immediately with empty outputs.
        for prompt, sp in zip(prompts, sparams):
            self.add_request(prompt, sp)

        # 改这里：outputs 用字典存储
        outputs: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()  # output: list of (seq_id, token_ids)
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            # 循环里保持原来的赋值
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 这里改：从字典取值
        outputs_list = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 解码成文本 + 保留 token_ids
        outputs_list = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs_list
        ]

        if use_tqdm:
            pbar.close()

        return outputs_list
