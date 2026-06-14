"""Generation helpers shared by NLA scripts."""


def decode_generated(output, prompt_tokens, tokenizer, stop_text="</explanation>"):
    """Extract generated text from a Transformers generate() output.

    When generation uses inputs_embeds, some Transformers versions return only
    generated token IDs while others return a prompt-prefixed sequence.
    We check whether the output starts with the prompt tokens to decide.

    prompt_tokens: list[int] or int (if int, treated as legacy prompt_len)
    """
    seq = output.sequences[0] if hasattr(output, "sequences") else output[0]

    if isinstance(prompt_tokens, int):
        prompt_len = prompt_tokens
        starts_with_prompt = (
            seq.shape[0] > prompt_len * 1.5
        )
    else:
        prompt_len = len(prompt_tokens)
        prefix = seq[:min(prompt_len, seq.shape[0])].tolist()
        starts_with_prompt = prefix == prompt_tokens[:len(prefix)] and seq.shape[0] > prompt_len

    gen_ids = seq[prompt_len:] if starts_with_prompt else seq

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if stop_text and stop_text in text:
        text = text.split(stop_text)[0]
    return text.strip()
