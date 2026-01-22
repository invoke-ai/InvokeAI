#!/usr/bin/env python3
"""Script to modify a user in the InvokeAI database.

This script provides a convenient way to modify user details (name, password, admin status)
in the InvokeAI database for administration purposes. It can be run from the command line
or imported and used programmatically.

Usage:
    # Interactive mode (prompts for details)
    python scripts/usermod.py

    # Command line mode
    python scripts/usermod.py --email user@example.com --name "New Name"
    python scripts/usermod.py --email user@example.com --password newpass123
    python scripts/usermod.py --email user@example.com --admin
    python scripts/usermod.py --email user@example.com --no-admin

Examples:
    # Change user's display name
    python scripts/usermod.py --email alice@test.local --name "Alice Johnson"

    # Change user's password
    python scripts/usermod.py --email alice@test.local --password NewPassword123

    # Make user an admin
    python scripts/usermod.py --email alice@test.local --admin

    # Remove admin privileges
    python scripts/usermod.py --email alice@test.local --no-admin

    # Change multiple properties at once
    python scripts/usermod.py --email alice@test.local --name "Alice Admin" --password Secret123 --admin
"""

import argparse
import getpass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def modify_user_interactive():
    """Modify a user interactively by prompting for details."""
    from invokeai.app.services.auth.password_utils import validate_password_strength
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_common import UserUpdateRequest
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    print("=== Modify InvokeAI User ===\n")

    # Get user email
    email = input("Email address of user to modify: ").strip()
    if not email:
        print("Error: Email is required")
        return False

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        # Get user to show current details
        user = user_service.get_by_email(email)
        if not user:
            print(f"\n❌ Error: No user found with email '{email}'")
            return False

        print(f"\nCurrent user details:")
        print(f"   User ID: {user.user_id}")
        print(f"   Email: {user.email}")
        print(f"   Display Name: {user.display_name or '(not set)'}")
        print(f"   Admin: {'Yes' if user.is_admin else 'No'}")
        print(f"   Active: {'Yes' if user.is_active else 'No'}")

        print("\n--- What would you like to change? (leave blank to keep current value) ---\n")

        # Get new display name
        new_name = input(f"New display name [{user.display_name or '(not set)'}]: ").strip()
        display_name = new_name if new_name else None

        # Get new password
        change_password = input("Change password? (y/N): ").strip().lower()
        password = None
        if change_password in ("y", "yes"):
            while True:
                password = getpass.getpass("New password: ")
                if not password:
                    print("Keeping existing password.")
                    password = None
                    break

                password_confirm = getpass.getpass("Confirm new password: ")

                if password != password_confirm:
                    print("Error: Passwords do not match. Please try again.\n")
                    continue

                # Validate password strength
                is_valid, error_msg = validate_password_strength(password)
                if not is_valid:
                    print(f"Error: {error_msg}\n")
                    continue

                break

        # Get new admin status
        change_admin = input("Change admin status? (y/N): ").strip().lower()
        is_admin = None
        if change_admin in ("y", "yes"):
            is_admin_input = (
                input(f"Make administrator? [current: {'Yes' if user.is_admin else 'No'}] (y/N): ").strip().lower()
            )
            is_admin = is_admin_input in ("y", "yes")

        # Check if any changes were made
        if display_name is None and password is None and is_admin is None:
            print("\nNo changes requested. User not modified.")
            return True

        # Update user
        changes = UserUpdateRequest(display_name=display_name, password=password, is_admin=is_admin)
        updated_user = user_service.update(user.user_id, changes)

        print("\n✅ User updated successfully!")
        print(f"   User ID: {updated_user.user_id}")
        print(f"   Email: {updated_user.email}")
        print(f"   Display Name: {updated_user.display_name or '(not set)'}")
        print(f"   Admin: {'Yes' if updated_user.is_admin else 'No'}")
        print(f"   Active: {'Yes' if updated_user.is_active else 'No'}")

        return True

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def modify_user_cli(
    email: str,
    display_name: str | None = None,
    password: str | None = None,
    is_admin: bool | None = None,
):
    """Modify a user via CLI arguments."""
    from invokeai.app.services.auth.password_utils import validate_password_strength
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_common import UserUpdateRequest
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    # Validate password if provided
    if password is not None:
        is_valid, error_msg = validate_password_strength(password)
        if not is_valid:
            print(f"❌ Password validation failed: {error_msg}")
            return False

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        # Get user to verify existence
        user = user_service.get_by_email(email)
        if not user:
            print(f"❌ Error: No user found with email '{email}'")
            return False

        # Check if any changes were requested
        if display_name is None and password is None and is_admin is None:
            print("❌ Error: No changes specified. Use --name, --password, --admin, or --no-admin")
            return False

        # Update user
        changes = UserUpdateRequest(display_name=display_name, password=password, is_admin=is_admin)
        updated_user = user_service.update(user.user_id, changes)

        print("✅ User updated successfully!")
        print(f"   User ID: {updated_user.user_id}")
        print(f"   Email: {updated_user.email}")
        print(f"   Display Name: {updated_user.display_name or '(not set)'}")
        print(f"   Admin: {'Yes' if updated_user.is_admin else 'No'}")
        print(f"   Active: {'Yes' if updated_user.is_active else 'No'}")

        return True

    except ValueError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Modify a user in the InvokeAI database",
        epilog="If no arguments are provided, the script will run in interactive mode.",
    )
    parser.add_argument("--email", "-e", help="User email address")
    parser.add_argument("--name", "-n", help="New display name")
    parser.add_argument("--password", "-p", help="New password")

    admin_group = parser.add_mutually_exclusive_group()
    admin_group.add_argument("--admin", "-a", action="store_true", help="Grant administrator privileges")
    admin_group.add_argument("--no-admin", dest="no_admin", action="store_true", help="Remove administrator privileges")

    args = parser.parse_args()

    # Determine admin status change
    is_admin = None
    if args.admin:
        is_admin = True
    elif args.no_admin:
        is_admin = False

    # Check if email was provided
    if args.email:
        # CLI mode
        success = modify_user_cli(args.email, args.name, args.password, is_admin)
    else:
        # Interactive mode
        success = modify_user_interactive()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
