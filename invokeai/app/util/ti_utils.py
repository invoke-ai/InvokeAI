import re


def extract_ti_triggers_from_prompt(prompt: str) -> list[str]:
    ti_triggers = []
    for trigger in re.findall(r"<[a-zA-Z0-9., _-]+>", prompt):
        ti_triggers.append(trigger)
    return ti_triggers
