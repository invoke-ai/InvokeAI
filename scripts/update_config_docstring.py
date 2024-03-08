import os

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

    from invokeai.app.services.config.config_default import InvokeAIAppConfig

    docstring = InvokeAIAppConfig.generate_docstrings()

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
