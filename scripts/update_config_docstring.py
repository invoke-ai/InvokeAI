import os
from typing import Literal, get_args, get_type_hints

from invokeai.app.services.config.config_default import InvokeAIAppConfig


def generate_config_docstrings() -> str:
    """Helper function for mkdocs. Generates a docstring for the InvokeAIAppConfig class.

    You shouldn't run this manually. Instead, run `scripts/update-config-docstring.py` to update the docstring.
    A makefile target is also available: `make update-config-docstring`.

    See that script for more information about why this is necessary.
    """
    docstring = '    """Invoke\'s global app configuration.\n\n'
    docstring += "    Typically, you won't need to interact with this class directly. Instead, use the `get_config` function from `invokeai.app.services.config` to get a singleton config object.\n\n"
    docstring += "    Attributes:\n"

    field_descriptions: list[str] = []
    type_hints = get_type_hints(InvokeAIAppConfig)

    for k, v in InvokeAIAppConfig.model_fields.items():
        if v.exclude:
            continue
        field_type = type_hints.get(k)
        extra = ""
        if getattr(field_type, "__origin__", None) is Literal:
            # Get options for literals - the docs generator can't pull these out
            options = [f"`{str(x)}`" for x in get_args(field_type)]
            extra = f"<br>Valid values: {', '.join(options)}"
        field_descriptions.append(f"        {k}: {v.description}{extra}")

    docstring += "\n".join(field_descriptions)
    docstring += '\n    """'

    return docstring


# The pydantic app config can be documented automatically using mkdocs, but this requires that the docstring
# for the class is kept up to date. We use a pydantic model for the app config. Each config setting is a field
# with a `description` parameter. It is tedious to update both the description _and_ the docstring for the class.
#
# This script parses the pydantic model, generates a valid docstring and replaces the existing docstring in the file,
# so you don't need to worry about keeping the docstring up to date.
#
# A test is provided to ensure that the docstring is up to date. If the test fails, run this script.


def main():
    # Change working directory to the repo root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    docstring = generate_config_docstrings()

    # Replace the docstring in the file
    with open("invokeai/app/services/config/config_default.py", "r") as f:
        lines = f.readlines()

    # Find the class definition line
    class_def_index = next(i for i, line in enumerate(lines) if "class InvokeAIAppConfig" in line)

    # Find the existing docstring start and end lines
    docstring_start_index = next(i for i, line in enumerate(lines[class_def_index:]) if '"""' in line) + class_def_index
    docstring_end_index = (
        next(i for i, line in enumerate(lines[docstring_start_index + 1 :]) if '"""' in line)
        + docstring_start_index
        + 1
    )

    # Replace the existing docstring with the new one, plus some line breaks in between. This _should_ result in a
    # correctly-formatted file with no syntax errors.
    lines = lines[:docstring_start_index] + [docstring, "\n"] + lines[docstring_end_index + 1 :]

    # Write the modified lines back to the file
    with open("invokeai/app/services/config/config_default.py", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
