import importlib.util
import inspect
import os


def load_custom_hooks(directory_path):
    hook_dict = {}

    for filename in os.listdir(directory_path):
        if not filename.endswith(".py"):
            continue

        module_name = filename[:-3]
        module_path = os.path.join(directory_path, filename)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        functions = {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction) if
                     name.startswith("hook_")}
        hook_dict.update(functions)

    return hook_dict


def get_position_list(position: str) -> list[int]:
    if position == "all":
        return []
    blocks = position.split(",")
    position_list = []
    for block in blocks:
        if "-" in block:
            start, end = block.split("-")
            position_list.extend(range(int(start), int(end) + 1))
        else:
            position_list.append(int(block))
    return position_list


def style_tokens(tokens: list[str], position_list: list[int]) -> str:
    styled_tokens = """<div style="display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap; border:1px solid grey; border-radius: 5px; padding: 10px 10px; margin-bottom: 10px;">"""
    for i, token in enumerate(tokens):
        clr = "#FFCCCB" if (not position_list or i in position_list) else "#90ee90"
        token_str = replace_chars(token, style_dict)
        styled_tokens += f"""<div style="text-align: center;"><div style="font-size: 16px; background-color:{clr};">{token_str}</div><div style="font-size: 14px; color: gray;">{i}</div></div>"""
    styled_tokens += "</div>"

    return styled_tokens


verbose_dict = {
    " ": "§SPACE§",
    "\n": "§NEWLINE§",
    "Ċ": "§NEWLINE§",
    "Ġ": "§SPACE§",
    "▁": "§SPACE§",
    "<0x0A>": "§NEWLINE§",
    "<|endoftext|>": "§ENDOFTEXT§",
}

style_dict = {
    "Ċ": "&nbsp;<br>",
    "Ġ": "&nbsp;",
    "▁": "&nbsp;",
    "<0x0A>": "&nbsp;<br>",
}


def replace_chars(text: str, replace_dict) -> str:
    for key, value in replace_dict.items():
        text = text.replace(key, value)
    return text


def format_output_token(tokens: list[str], top_probabilities: list[tuple[str, float]], replace_dict: dict) -> str:
    html_content = ""
    for j, (token, probs) in enumerate(zip(tokens, top_probabilities)):
        if not probs:
            continue
        token_str = replace_dict[token] if token in replace_dict else token
        token_str = replace_chars(token_str, style_dict)
        prob_text = "<table><tr><th>Token</th><th>Probability</th></tr>" + "".join(
            [
                f'<tr class="tooltip-item"><td>{verbose_dict.get(p[0], p[0])}</td><td>{p[1]:.2f}</td></tr>'
                for p in probs]
        ) + "</table>"
        html_content += f"""<div class="hover-container">{token_str}<div class="tooltip">{prob_text}</div></div>"""

    return html_content
