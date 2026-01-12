#!/usr/bin/env python3
"""Script to add a user to the InvokeAI database.

This script provides a convenient way to add users (admin or regular) to the InvokeAI
database for testing and administration purposes. It can be run from the command line
or imported and used programmatically.

Usage:
    # Interactive mode (prompts for all details)
    python scripts/add_user.py

    # Command line mode
    python scripts/add_user.py --email user@example.com --password securepass123 --name "Test User"

    # Add admin user
    python scripts/add_user.py --email admin@example.com --password adminpass123 --admin

Examples:
    # Add a regular user
    python scripts/add_user.py --email alice@test.local --password Password123 --name "Alice Smith"

    # Add an admin user
    python scripts/add_user.py --email admin@test.local --password AdminPass123 --name "Admin User" --admin
"""

import argparse
import getpass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def add_user_interactive():
    """Add a user interactively by prompting for details."""
    from invokeai.app.services.auth.password_utils import validate_password_strength
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_common import UserCreateRequest
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    print("=== Add InvokeAI User ===\n")

    # Get user details
    email = input("Email address: ").strip()
    if not email:
        print("Error: Email is required")
        return False

    display_name = input("Display name (optional): ").strip() or None

    # Get password with confirmation
    while True:
        password = getpass.getpass("Password: ")
        password_confirm = getpass.getpass("Confirm password: ")

        if password != password_confirm:
            print("Error: Passwords do not match. Please try again.\n")
            continue

        # Validate password strength
        is_valid, error_msg = validate_password_strength(password)
        if not is_valid:
            print(f"Error: {error_msg}\n")
            continue

        break

    # Ask if user should be admin
    is_admin_input = input("Make this user an administrator? (y/N): ").strip().lower()
    is_admin = is_admin_input in ("y", "yes")

    # Create user
    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        user_data = UserCreateRequest(email=email, display_name=display_name, password=password, is_admin=is_admin)

        user = user_service.create(user_data)

        print("\n✅ User created successfully!")
        print(f"   User ID: {user.user_id}")
        print(f"   Email: {user.email}")
        print(f"   Display Name: {user.display_name or '(not set)'}")
        print(f"   Admin: {'Yes' if user.is_admin else 'No'}")
        print(f"   Active: {'Yes' if user.is_active else 'No'}")

        return True

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def add_user_cli(email: str, password: str, display_name: str | None = None, is_admin: bool = False):
    """Add a user via CLI arguments."""
    from invokeai.app.services.auth.password_utils import validate_password_strength
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_common import UserCreateRequest
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    # Validate password
    is_valid, error_msg = validate_password_strength(password)
    if not is_valid:
        print(f"❌ Password validation failed: {error_msg}")
        return False

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        user_data = UserCreateRequest(email=email, display_name=display_name, password=password, is_admin=is_admin)

        user = user_service.create(user_data)

        print("✅ User created successfully!")
        print(f"   User ID: {user.user_id}")
        print(f"   Email: {user.email}")
        print(f"   Display Name: {user.display_name or '(not set)'}")
        print(f"   Admin: {'Yes' if user.is_admin else 'No'}")
        print(f"   Active: {'Yes' if user.is_active else 'No'}")

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
        description="Add a user to the InvokeAI database",
        epilog="If no arguments are provided, the script will run in interactive mode.",
    )
    parser.add_argument("--email", "-e", help="User email address")
    parser.add_argument("--password", "-p", help="User password")
    parser.add_argument("--name", "-n", help="User display name (optional)")
    parser.add_argument("--admin", "-a", action="store_true", help="Make user an administrator")

    args = parser.parse_args()

    # Check if any arguments were provided
    if args.email or args.password:
        # CLI mode - require both email and password
        if not args.email or not args.password:
            print("❌ Error: Both --email and --password are required when using CLI mode")
            print("   Run without arguments for interactive mode")
            sys.exit(1)

        success = add_user_cli(args.email, args.password, args.name, args.admin)
    else:
        # Interactive mode
        success = add_user_interactive()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
