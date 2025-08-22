import tiktoken
import uuid

from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

MAX_INPUT_TOKENS = 4096
enc = tiktoken.get_encoding("cl100k_base")

MESSAGES_ROLES = ["user", "system", "assistant"]

START_USER = "<start_of_turn>user"
START_MODEL = "<start_of_turn>model"
END_TURN = "<end_of_turn>"

def format_messages_for_gemma(messages: list[dict]) -> str:
    formatted = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system" or role == "user":
            formatted += f"{START_USER}\n{content}\n{END_TURN}"
        elif role == "assistant":
            formatted += f"{START_MODEL}\n{content}\n{END_TURN}"
    formatted += START_MODEL
    return formatted

def init_engine() -> AsyncLLM:
    engine_args = AsyncEngineArgs(
        model="models/gemma-3-270m-it",
        served_model_name="Marshall-gemma3-270m",
        enforce_eager=False,
        gpu_memory_utilization=0.75,
        max_seq_len_to_capture=8192,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        swap_space=2,
        seed=777,
        max_num_seqs=3,
        enable_prefix_caching=True,
        disable_log_stats=False
    )
    return AsyncLLM.from_engine_args(engine_args)

def validate_messages(messages: list[dict]) -> None:
    if messages[0]["role"] != "system":
        raise ValueError("First message must be a system message")
    if messages[1]["role"] != "user":
        raise ValueError("Second message must be a user message")
    for i, message in enumerate(messages):
        if message["role"] not in MESSAGES_ROLES:
            raise ValueError(
                f"Invalid role on message {i}: {message['role']}. Must be one of {MESSAGES_ROLES}"
            )

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def validate_len_token(messages: list[dict], max_tokens: int) -> None:
    prompt = format_messages_for_gemma(messages)
    if count_tokens(prompt) + max_tokens > MAX_INPUT_TOKENS:
        raise ValueError("Total message length exceeds max token limit")

async def generate_chat_stream(engine, messages, temperature, max_tokens):
    validate_messages(messages)
    validate_len_token(messages, max_tokens)
    
    async for output in engine.generate(
        request_id=f"chat-stream-{uuid.uuid4()}",
        prompt=format_messages_for_gemma(messages),
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            output_kind=RequestOutputKind.DELTA,
            seed=777
        ),
    ):
        for completion in output.outputs:
            if completion.text:
                yield f"data: {completion.text}\n\n"

        if output.finished:
            return


async def generate_chat_once(engine: AsyncLLM,
                             messages: list[dict],
                             temperature: float = 0.7,
                             max_tokens: int = 1024) -> str:
    validate_messages(messages)
    validate_len_token(messages, max_tokens)

    prompt = format_messages_for_gemma(messages)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        output_kind=RequestOutputKind.FINAL_ONLY,
        seed=777
    )

    request_id = f"chat-{uuid.uuid4()}"

    async for output in engine.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params,
    ):
        if output.finished:
            return output.outputs[0].text.strip()

