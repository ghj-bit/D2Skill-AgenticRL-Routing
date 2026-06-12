import re
from typing import List


def _normalize_action(action: str) -> str:
    action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
    return " ".join(action.split()).strip().lower()


def _is_valid_textcraft_action(action: str) -> bool:
    return bool(
        action == "inventory"
        or re.match(r"^get [0-9]+ .+", action)
        or re.match(r"^craft (.+) using (.+)", action)
    )


def _parse_tagged_action(text: str):
    stripped = text.strip()
    action = ""
    valid = 0
    match = re.fullmatch(
        r"<think>\s*(.*?)\s*</think>\s*<action>\s*(.*?)\s*</action>",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        reasoning = match.group(1).strip()
        action = match.group(2).strip()
        valid = int(bool(reasoning) and bool(action))
    else:
        action_matches = re.findall(
            r"<action>\s*(.*?)\s*</action>",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if len(action_matches) == 1:
            action = action_matches[0].strip()
    return action, valid


def _parse_agentgym_action(text: str):
    action_matches = re.findall(
        r"(?:^|\n)\s*Action:\s*(.*?)(?=\n\s*(?:Thought|Action):|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    thought_matches = re.findall(
        r"(?:^|\n)\s*Thought:\s*(.*?)(?=\n\s*Action:|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if len(action_matches) != 1:
        return "", 0
    action = action_matches[0].strip()
    valid = int(len(thought_matches) == 1 and bool(thought_matches[0].strip()) and bool(action))
    return action, valid


def textcraft_projection(actions: List[str], output_format: str = "agentgym"):
    output_format = str(output_format or "agentgym").strip().lower()
    valids = [0] * len(actions)
    projected = []

    for original in actions:
        text = str(original or "")
        if output_format == "tagged":
            action, valid = _parse_tagged_action(text)
        else:
            action, valid = _parse_agentgym_action(text)

        valids[len(projected)] = valid
        action = _normalize_action(action)
        if not _is_valid_textcraft_action(action):
            valids[len(projected)] = 0
        if re.search(r"[\u4e00-\u9fff]", text):
            valids[len(projected)] = 0

        projected.append(action)

    return projected, valids
