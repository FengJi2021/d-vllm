import os
import logging
import logging.config
from transformers import AutoTokenizer
from dvllm import LLM, SParams

def setup_logger(level: str = "DEBUG", log_file: str="test_run.log"):
    LOGGER_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "default",
                "level": level
            }
        },
        "loggers": {
            # root logger
            "": {"handlers": ["console"], "level": level},
            # dvllm logger
            "dvllm": {
                "handlers": ["console", "file"],
                "level": level,
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(LOGGER_CONFIG)
    

def main():
    setup_logger()

    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    logging.info(f"Loading model from: {model_path}")

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 初始化 LLM，指定 MPS
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        device_type="mps",
    )

    # 生成参数
    sparams = SParams(temperature=0.7, max_tokens=128)

    # 一定要使用 chat template
    prompts = [
        "<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n"
    ]

    logging.debug(f"Prompts for model: {prompts}")

    # 调用 generate
    outputs = llm.generate(prompts, sparams)

    # 输出结果
    for prompt, output in zip(prompts, outputs):
        logging.info("\n==============================")
        logging.info(f"Prompt:\n{prompt}")
        logging.info(f"Completion:\n{output['text']}")
        logging.info("==============================\n")

if __name__ == "__main__":
    main()
