import huggingface_hub
from pwinput import pwinput

from invokeai.app.util.suppress_output import SuppressOutput


def hf_login() -> None:
    """Prompts the user for their HuggingFace token. If a valid token is already saved, this function will do nothing.

    Returns:
        bool: True if the login was successful, False if the user canceled.

    Raises:
        RuntimeError: If the user cancels the login prompt.
    """

    current_token = huggingface_hub.get_token()

    try:
        if huggingface_hub.get_token_permission(current_token):
            # We have a valid token already
            return
    except ConnectionError:
        print("Unable to reach HF to verify token. Skipping...")
        # No internet connection, so we can't check the token
        pass

    # InvokeAILogger depends on the config, and this class is used within the config, so we can't use the app logger here
    print("Enter your HuggingFace token. This is required to convert checkpoint/safetensors models to diffusers.")
    print("For more information, see https://huggingface.co/docs/hub/security-tokens#how-to-manage-user-access-tokens")
    print("Press Ctrl+C to skip.")

    while True:
        try:
            access_token = pwinput(prompt="HF token: ")
            # The login function prints to stdout
            with SuppressOutput():
                huggingface_hub.login(token=access_token, add_to_git_credential=False)
            print("Token verified.")
            break
        except ValueError:
            print("Invalid token!")
            continue
        except KeyboardInterrupt:
            print("\nToken verification canceled.")
            break
