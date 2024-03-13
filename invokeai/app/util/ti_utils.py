import re
from typing import List, Tuple

import invokeai.backend.util.logging as logger
from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import BaseModelType, ModelType
from invokeai.backend.textual_inversion import TextualInversionModelRaw


def extract_ti_triggers_from_prompt(prompt: str) -> List[str]:
    ti_triggers: List[str] = []
    for trigger in re.findall(r"<[a-zA-Z0-9., _-]+>", prompt):
        ti_triggers.append(str(trigger))
    return ti_triggers


def generate_ti_list(
    prompt: str, base: BaseModelType, context: InvocationContext
) -> List[Tuple[str, TextualInversionModelRaw]]:
    ti_list: List[Tuple[str, TextualInversionModelRaw]] = []
    for trigger in extract_ti_triggers_from_prompt(prompt):
        name_or_key = trigger[1:-1]
        try:
            loaded_model = context.models.load(name_or_key)
            model = loaded_model.model
            assert isinstance(model, TextualInversionModelRaw)
            assert loaded_model.config.base == base
            ti_list.append((name_or_key, model))
        except UnknownModelException:
            try:
                loaded_model = context.models.load_by_attrs(
                    name=name_or_key, base=base, type=ModelType.TextualInversion
                )
                model = loaded_model.model
                assert isinstance(model, TextualInversionModelRaw)
                assert loaded_model.config.base == base
                ti_list.append((name_or_key, model))
            except UnknownModelException:
                pass
        except ValueError:
            logger.warning(f'trigger: "{trigger}" more than one similarly-named textual inversion models')
        except AssertionError:
            logger.warning(f'trigger: "{trigger}" not a valid textual inversion model for this graph')
        except Exception:
            logger.warning(f'Failed to load TI model for trigger: "{trigger}"')
    return ti_list
