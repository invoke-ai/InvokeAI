import re
from typing import List


def extract_ti_triggers_from_prompt(prompt: str) -> List[str]:
    ti_triggers: List[str] = []
    for trigger in re.findall(r"<[a-zA-Z0-9., _-]+>", prompt):
        ti_triggers.append(str(trigger))
    return ti_triggers
