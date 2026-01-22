#!/usr/bin/env python3
"""Script to list users from the InvokeAI database.

This script provides a convenient way to view all users in the InvokeAI database
with their details. It can output in table format (default) or JSON format.

Usage:
    # Display users as a table
    python scripts/userlist.py

    # Display users as JSON
    python scripts/userlist.py --json

Examples:
    # View all users in table format
    python scripts/userlist.py

    # View all users in JSON format for scripting
    python scripts/userlist.py --json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def list_users_table():
    """List all users in a formatted table."""
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    config = get_config()
    logger = InvokeAILogger.get_logger(config=config)
    db = SqliteDatabase(config.db_path, logger)
    user_service = UserService(db)

    try:
        # Get all users
        users = user_service.list_users()

        if not users:
            print("No users found in database.")
            return True

        # Print header
        print("\n=== InvokeAI Users ===\n")
        print(
            f"{'User ID':<36} {'Email':<30} {'Display Name':<20} {'Admin':<8} {'Active':<8}"
        )
        print("-" * 108)

        # Print each user
        for user in users:
            user_id = user.user_id
            email = user.email[:29] if len(user.email) > 29 else user.email
            name = user.display_name[:19] if len(user.display_name) > 19 else user.display_name
            is_admin = "Yes" if user.is_admin else "No"
            is_active = "Yes" if user.is_active else "No"

            print(f"{user_id:<36} {email:<30} {name:<20} {is_admin:<8} {is_active:<8}")

        print(f"\nTotal users: {len(users)}")
        return True

    except Exception as e:
        print(f"Error listing users: {e}")
        return False


def list_users_json():
    """List all users in JSON format."""
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    config = get_config()
    logger = InvokeAILogger.get_logger(config=config)
    db = SqliteDatabase(config.db_path, logger)
    user_service = UserService(db)

    try:
        # Get all users
        users = user_service.list_users()

        # Convert to JSON-serializable format
        users_data = [
            {
                "id": user.user_id,
                "email": user.email,
                "name": user.display_name,
                "is_admin": user.is_admin,
                "is_active": user.is_active,
            }
            for user in users
        ]

        # Print JSON
        print(json.dumps(users_data, indent=2))
        return True

    except Exception as e:
        print(f'{{"error": "{e}"}}', file=sys.stderr)
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="List users from the InvokeAI database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all users in table format
  python scripts/userlist.py

  # View all users in JSON format for scripting
  python scripts/userlist.py --json
        """,
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output users in JSON format instead of table",
    )

    args = parser.parse_args()

    # List users in requested format
    if args.json:
        success = list_users_json()
    else:
        success = list_users_table()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
