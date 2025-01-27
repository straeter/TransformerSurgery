

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
    # Create HTML for tokens with positions below
    styled_tokens = """<div style="display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap; border:1px solid grey; border-radius: 5px; padding: 10px 10px; margin-bottom: 10px;">"""
    for i, token in enumerate(tokens):
        clr = "#FFCCCB" if (not position_list or i in position_list) else "#90ee90"
        styled_tokens += f"""<div style="text-align: center;"><div style="font-size: 16px; background-color:{clr};">{token}</div><div style="font-size: 14px; color: gray;">{i}</div></div>"""
    styled_tokens += "</div>"

    return styled_tokens

    # full_html = f"""
    # <div style="display: block;">{styled_tokens}</div>
    # """
    #
    # return full_html
