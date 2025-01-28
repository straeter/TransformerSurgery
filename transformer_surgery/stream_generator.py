from copy import deepcopy

import numpy as np
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


def get_top_tokens(tokenizer, probabilities, n_p):
    top_probabilities, top_indices = torch.topk(probabilities, n_p, dim=-1)
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices[0]]
    return list(zip(top_tokens, top_probabilities[0].tolist()))


def generate_with_hooks(model, prompt: str, hooks: list, max_tokens: int, temperature: float = 0.0, top_k: int = None,
                        top_p: float = None, n_p: int = 10):

    max_tokens = max_tokens if max_tokens > 0 else 100000

    # Tokenize the prompt
    tokens = model.to_tokens(prompt)
    generated_tokens = tokens.clone()
    top_probabilities = [[] for _ in range(len(tokens[0]))]

    for _ in range(max_tokens):
        # Run the model with hooks
        with model.hooks(fwd_hooks=hooks):
            logits = model(generated_tokens)

        # Extract the logits for the last token
        next_token_logits = logits[:, -1, :]

        # Compute the probabilities + top probabilities (for the visualization)
        probabilities = torch.softmax(next_token_logits, dim=-1)
        top_probabilities.append(get_top_tokens(model.tokenizer, probabilities, n_p))

        if temperature == 0:
            # Temperature 0: Greedy sampling (no scaling)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            if top_k:
                probabilities = apply_top_k(probabilities, top_k)

            if top_p:
                probabilities = apply_top_p(probabilities, top_p)

            # Sample from the probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1)

        # Append the generated token
        generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

        # Yield generated tokens and top probabilities
        yield model.tokenizer.convert_ids_to_tokens(generated_tokens[0]), top_probabilities

        # Stop if the model outputs the end-of-sequence token
        if next_token.item() == model.tokenizer.eos_token_id:
            break

    yield model.tokenizer.convert_ids_to_tokens(generated_tokens[0]) + [" $MAX TOKENS REACHED$"], top_probabilities + [[]]

    # Clean up to release gpu memory
    for var_name in list(locals().keys()):
        del locals()[var_name]
    torch.cuda.empty_cache()
