import re
from typing import List


def textcraft_projection(actions: List[str]):
    valids = [0] * len(actions)
    projected = []

    for original in actions:
        text = str(original or "")
        stripped = text.strip()
        action = ""
        match = re.fullmatch(
            r"<think>\s*(.*?)\s*</think>\s*<action>\s*(.*?)\s*</action>",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            reasoning = match.group(1).strip()
            action = match.group(2).strip()
            valids[len(projected)] = int(bool(reasoning) and bool(action))

        action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
        action = " ".join(action.split()).strip().lower()
        if not (
            action == "inventory"
            or re.match(r"^get [0-9]+ .+", action)
            or re.match(r"^craft (.+) using (.+)", action)
        ):
            valids[len(projected)] = 0
        if re.search(r"[\u4e00-\u9fff]", text):
            valids[len(projected)] = 0

        projected.append(action)

    return projected, valids
