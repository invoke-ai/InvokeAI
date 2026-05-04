from pathlib import PureWindowsPath


# TODO: Should these excpetions subclass existing python exceptions?
class ImageFileNotFoundException(Exception):
    """Raised when an image file is not found in storage."""

    def __init__(self, message="Image file not found"):
        super().__init__(message)


class ImageFileSaveException(Exception):
    """Raised when an image cannot be saved."""

    def __init__(self, message="Image file not saved"):
        super().__init__(message)


class ImageFileDeleteException(Exception):
    """Raised when an image cannot be deleted."""

    def __init__(self, message="Image file not deleted"):
        super().__init__(message)


def validate_subfolder(subfolder: str) -> None:
    """Validate a subfolder path to prevent directory traversal while allowing controlled subdirectories.

    Shared by the disk and S3 backends so both accept exactly the same set of subfolder values.
    """
    if not subfolder:
        return
    if "\\" in subfolder:
        raise ValueError("Backslashes not allowed in subfolder path")
    if subfolder.startswith("/"):
        raise ValueError("Absolute paths not allowed in subfolder path")
    # Reject Windows drive-qualified / absolute forms ("C:/x", UNC) OS-agnostically:
    # disk rejects them via is_relative_to() on Windows, but S3 would create literal
    # "C:/..." keys. PureWindowsPath parses them regardless of the host OS.
    windows_form = PureWindowsPath(subfolder)
    if windows_form.drive or windows_form.is_absolute():
        raise ValueError("Drive-qualified or absolute paths not allowed in subfolder path")
    for part in subfolder.split("/"):
        if part == "..":
            raise ValueError("Parent directory references not allowed in subfolder path")
        if part == ".":
            # Harmless on disk (Path.resolve normalizes it away) but preserved in S3
            # keys, so reject it to keep the two backends' object paths identical.
            raise ValueError("Current directory references not allowed in subfolder path")
        if part == "":
            raise ValueError("Empty path segments not allowed in subfolder path")
