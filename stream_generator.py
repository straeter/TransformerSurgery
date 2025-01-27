from copy import deepcopy

import torch


def apply_top_k(probabilities: torch.Tensor, top_k: int):
    sorted_prob, sorted_indices = torch.sort(probabilities, descending=True)
    probabilities[probabilities < sorted_prob[0, top_k]] = 0
    probabilities = probabilities / torch.sum(probabilities, dim=-1, keepdim=True)
    return probabilities


def apply_top_p(probabilities: torch.Tensor, top_p: float):
    sorted_prob, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_prob, dim=-1)
    keep_tokens = cumulative_probabilities < top_p
    probabilities[keep_tokens] = 0
    probabilities = probabilities / torch.sum(probabilities, dim=-1, keepdim=True)
    return probabilities


def generate_with_hooks(model, prompt: str, hooks: list, max_tokens: int, temperature: float = 0.0, top_k: int = None,
                        top_p: float = None):
    # Tokenize the prompt
    tokens = model.to_tokens(prompt)
    generated_tokens = tokens.clone()

    max_tokens = max_tokens if max_tokens > 0 else 100000

    # Generate token by token
    for _ in range(max_tokens):
        # Run the model with hooks
        with model.hooks(fwd_hooks=hooks):
            logits = model(generated_tokens)

        # Extract the logits for the last token
        next_token_logits = logits[:, -1, :]

        if temperature == 0:
            # Temperature 0: Greedy sampling (no scaling)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)

            if top_k:
                probabilities = apply_top_k(probabilities, top_k)

            if top_p:
                probabilities = apply_top_p(probabilities, top_p)

            # Sample from the probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1)

        # Append the generated token
        generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

        # Yield next token
        yield model.tokenizer.decode(generated_tokens[0])

        # Stop if the model outputs the end-of-sequence token
        if next_token.item() == model.tokenizer.eos_token_id:
            break

    yield model.tokenizer.decode(generated_tokens[0]) + " <MAX TOKENS REACHED>"

    for var_name in list(locals().keys()):
        del locals()[var_name]
