from copy import deepcopy

import torch


def generate_with_hooks(model, prompt: str,hooks: list, max_tokens: int, temperature: float = 0.0):
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
