#!/usr/bin/env python3
"""Script to delete a user from the InvokeAI database.

This script provides a convenient way to delete users from the InvokeAI database
for administration purposes. It can be run from the command line or imported and
used programmatically.

Usage:
    # Interactive mode (prompts for email)
    python scripts/userdel.py

    # Command line mode
    python scripts/userdel.py --email user@example.com

    # Force delete without confirmation
    python scripts/userdel.py --email user@example.com --force

Examples:
    # Delete a user with confirmation
    python scripts/userdel.py --email alice@test.local

    # Delete a user without confirmation prompt
    python scripts/userdel.py --email alice@test.local --force
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def delete_user_interactive():
    """Delete a user interactively by prompting for email."""
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    print("=== Delete InvokeAI User ===\n")

    # Get user email
    email = input("Email address of user to delete: ").strip()
    if not email:
        print("Error: Email is required")
        return False

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        # Get user to show details before deletion
        user = user_service.get_by_email(email)
        if not user:
            print(f"\n❌ Error: No user found with email '{email}'")
            return False

        print(f"\nUser to delete:")
        print(f"   User ID: {user.user_id}")
        print(f"   Email: {user.email}")
        print(f"   Display Name: {user.display_name or '(not set)'}")
        print(f"   Admin: {'Yes' if user.is_admin else 'No'}")
        print(f"   Active: {'Yes' if user.is_active else 'No'}")

        # Confirm deletion
        confirm = input("\n⚠️  Are you sure you want to delete this user? (yes/no): ").strip().lower()
        if confirm not in ("yes", "y"):
            print("Deletion cancelled.")
            return False

        user_service.delete(user.user_id)

        print("\n✅ User deleted successfully!")
        return True

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def delete_user_cli(email: str, force: bool = False):
    """Delete a user via CLI arguments."""
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        # Get user to show details before deletion
        user = user_service.get_by_email(email)
        if not user:
            print(f"❌ Error: No user found with email '{email}'")
            return False

        if not force:
            print(f"User to delete:")
            print(f"   User ID: {user.user_id}")
            print(f"   Email: {user.email}")
            print(f"   Display Name: {user.display_name or '(not set)'}")
            print(f"   Admin: {'Yes' if user.is_admin else 'No'}")
            print(f"   Active: {'Yes' if user.is_active else 'No'}")

            confirm = input("\n⚠️  Are you sure you want to delete this user? (yes/no): ").strip().lower()
            if confirm not in ("yes", "y"):
                print("Deletion cancelled.")
                return False

        user_service.delete(user.user_id)

        print("✅ User deleted successfully!")
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
        description="Delete a user from the InvokeAI database",
        epilog="If no arguments are provided, the script will run in interactive mode.",
    )
    parser.add_argument("--email", "-e", help="User email address")
    parser.add_argument("--force", "-f", action="store_true", help="Delete without confirmation prompt")

    args = parser.parse_args()

    # Check if email was provided
    if args.email:
        # CLI mode
        success = delete_user_cli(args.email, args.force)
    else:
        # Interactive mode
        success = delete_user_interactive()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
