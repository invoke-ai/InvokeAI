"""User management command entry points for InvokeAI.

These functions are registered as console scripts in pyproject.toml and can be
called from the command line after installing the package:

    invoke-useradd   -- add a user
    invoke-userdel   -- delete a user
    invoke-userlist  -- list users
    invoke-usermod   -- modify a user
"""

import argparse
import getpass
import json
import os
import sys

_root_help = (
    "Path to the InvokeAI root directory. If omitted, the root is resolved in this order: "
    "the $INVOKEAI_ROOT environment variable, the active virtual environment's parent directory, "
    "or $HOME/invokeai."
)

# ---------------------------------------------------------------------------
# useradd
# ---------------------------------------------------------------------------


def _add_user_interactive() -> bool:
    """Add a user interactively by prompting for details."""
    from invokeai.app.services.auth.password_utils import validate_password_strength
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_common import UserCreateRequest
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    print("=== Add InvokeAI User ===\n")

    email = input("Email address: ").strip()
    if not email:
        print("Error: Email is required")
        return False

    display_name = input("Display name (optional): ").strip() or None

    while True:
        password = getpass.getpass("Password: ")
        password_confirm = getpass.getpass("Confirm password: ")

        if password != password_confirm:
            print("Error: Passwords do not match. Please try again.\n")
            continue

        is_valid, error_msg = validate_password_strength(password)
        if not is_valid:
            print(f"Error: {error_msg}\n")
            continue

        break

    is_admin_input = input("Make this user an administrator? (y/N): ").strip().lower()
    is_admin = is_admin_input in ("y", "yes")

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


def _add_user_cli(email: str, password: str, display_name: str | None = None, is_admin: bool = False) -> bool:
    """Add a user via CLI arguments."""
    from invokeai.app.services.auth.password_utils import validate_password_strength
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_common import UserCreateRequest
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

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


def useradd() -> None:
    """Entry point for ``invoke-useradd``."""
    parser = argparse.ArgumentParser(
        description="Add a user to the InvokeAI database",
        epilog="If no arguments are provided, the script will run in interactive mode.",
    )
    parser.add_argument("--root", "-r", help=_root_help)
    parser.add_argument("--email", "-e", help="User email address")
    parser.add_argument("--password", "-p", help="User password")
    parser.add_argument("--name", "-n", help="User display name (optional)")
    parser.add_argument("--admin", "-a", action="store_true", help="Make user an administrator")

    args = parser.parse_args()

    if args.root:
        os.environ["INVOKEAI_ROOT"] = args.root

    if args.email or args.password:
        if not args.email or not args.password:
            print("❌ Error: Both --email and --password are required when using CLI mode")
            print("   Run without arguments for interactive mode")
            sys.exit(1)
        success = _add_user_cli(args.email, args.password, args.name, args.admin)
    else:
        success = _add_user_interactive()

    sys.exit(0 if success else 1)


# ---------------------------------------------------------------------------
# userdel
# ---------------------------------------------------------------------------


def _delete_user_interactive() -> bool:
    """Delete a user interactively by prompting for email."""
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    print("=== Delete InvokeAI User ===\n")

    email = input("Email address of user to delete: ").strip()
    if not email:
        print("Error: Email is required")
        return False

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        user = user_service.get_by_email(email)
        if not user:
            print(f"\n❌ Error: No user found with email '{email}'")
            return False

        print("\nUser to delete:")
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


def _delete_user_cli(email: str, force: bool = False) -> bool:
    """Delete a user via CLI arguments."""
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        user = user_service.get_by_email(email)
        if not user:
            print(f"❌ Error: No user found with email '{email}'")
            return False

        if not force:
            print("User to delete:")
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


def userdel() -> None:
    """Entry point for ``invoke-userdel``."""
    parser = argparse.ArgumentParser(
        description="Delete a user from the InvokeAI database",
        epilog="If no arguments are provided, the script will run in interactive mode.",
    )
    parser.add_argument("--root", "-r", help=_root_help)
    parser.add_argument("--email", "-e", help="User email address")
    parser.add_argument("--force", "-f", action="store_true", help="Delete without confirmation prompt")

    args = parser.parse_args()

    if args.root:
        os.environ["INVOKEAI_ROOT"] = args.root

    if args.email:
        success = _delete_user_cli(args.email, args.force)
    else:
        success = _delete_user_interactive()

    sys.exit(0 if success else 1)


# ---------------------------------------------------------------------------
# userlist
# ---------------------------------------------------------------------------


def _list_users_table() -> bool:
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
        users = user_service.list_users()

        if not users:
            print("No users found in database.")
            return True

        print("\n=== InvokeAI Users ===\n")
        print(f"{'User ID':<36} {'Email':<30} {'Display Name':<20} {'Admin':<8} {'Active':<8}")
        print("-" * 108)

        for user in users:
            user_id = user.user_id
            email = user.email[:29] if len(user.email) > 29 else user.email
            raw_name = user.display_name or ""
            name = raw_name[:19] if len(raw_name) > 19 else raw_name
            is_admin = "Yes" if user.is_admin else "No"
            is_active = "Yes" if user.is_active else "No"
            print(f"{user_id:<36} {email:<30} {name:<20} {is_admin:<8} {is_active:<8}")

        print(f"\nTotal users: {len(users)}")
        return True

    except Exception as e:
        print(f"Error listing users: {e}")
        return False


def _list_users_json() -> bool:
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
        users = user_service.list_users()

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

        print(json.dumps(users_data, indent=2))
        return True

    except Exception as e:
        print(f'{{"error": "{e}"}}', file=sys.stderr)
        return False


def userlist() -> None:
    """Entry point for ``invoke-userlist``."""
    parser = argparse.ArgumentParser(
        description="List users from the InvokeAI database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  invoke-userlist
  invoke-userlist --json
        """,
    )
    parser.add_argument("--root", "-r", help=_root_help)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output users in JSON format instead of table",
    )

    args = parser.parse_args()

    if args.root:
        os.environ["INVOKEAI_ROOT"] = args.root

    success = _list_users_json() if args.json else _list_users_table()
    sys.exit(0 if success else 1)


# ---------------------------------------------------------------------------
# usermod
# ---------------------------------------------------------------------------


def _modify_user_interactive() -> bool:
    """Modify a user interactively by prompting for details."""
    from invokeai.app.services.auth.password_utils import validate_password_strength
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_common import UserUpdateRequest
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    print("=== Modify InvokeAI User ===\n")

    email = input("Email address of user to modify: ").strip()
    if not email:
        print("Error: Email is required")
        return False

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        user = user_service.get_by_email(email)
        if not user:
            print(f"\n❌ Error: No user found with email '{email}'")
            return False

        print("\nCurrent user details:")
        print(f"   User ID: {user.user_id}")
        print(f"   Email: {user.email}")
        print(f"   Display Name: {user.display_name or '(not set)'}")
        print(f"   Admin: {'Yes' if user.is_admin else 'No'}")
        print(f"   Active: {'Yes' if user.is_active else 'No'}")

        print("\n--- What would you like to change? (leave blank to keep current value) ---\n")

        new_name = input(f"New display name [{user.display_name or '(not set)'}]: ").strip()
        display_name = new_name if new_name else None

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

                is_valid, error_msg = validate_password_strength(password)
                if not is_valid:
                    print(f"Error: {error_msg}\n")
                    continue

                break

        change_admin = input("Change admin status? (y/N): ").strip().lower()
        is_admin = None
        if change_admin in ("y", "yes"):
            is_admin_input = (
                input(f"Make administrator? [current: {'Yes' if user.is_admin else 'No'}] (y/N): ").strip().lower()
            )
            is_admin = is_admin_input in ("y", "yes")

        if display_name is None and password is None and is_admin is None:
            print("\nNo changes requested. User not modified.")
            return True

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


def _modify_user_cli(
    email: str,
    display_name: str | None = None,
    password: str | None = None,
    is_admin: bool | None = None,
) -> bool:
    """Modify a user via CLI arguments."""
    from invokeai.app.services.auth.password_utils import validate_password_strength
    from invokeai.app.services.config import get_config
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.users.users_common import UserUpdateRequest
    from invokeai.app.services.users.users_default import UserService
    from invokeai.backend.util.logging import InvokeAILogger

    if password is not None:
        is_valid, error_msg = validate_password_strength(password)
        if not is_valid:
            print(f"❌ Password validation failed: {error_msg}")
            return False

    try:
        config = get_config()
        db = SqliteDatabase(config.db_path, InvokeAILogger.get_logger())
        user_service = UserService(db)

        user = user_service.get_by_email(email)
        if not user:
            print(f"❌ Error: No user found with email '{email}'")
            return False

        if display_name is None and password is None and is_admin is None:
            print("❌ Error: No changes specified. Use --name, --password, --admin, or --no-admin")
            return False

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


def usermod() -> None:
    """Entry point for ``invoke-usermod``."""
    parser = argparse.ArgumentParser(
        description="Modify a user in the InvokeAI database",
        epilog="If no arguments are provided, the script will run in interactive mode.",
    )
    parser.add_argument("--root", "-r", help=_root_help)
    parser.add_argument("--email", "-e", help="User email address")
    parser.add_argument("--name", "-n", help="New display name")
    parser.add_argument("--password", "-p", help="New password")

    admin_group = parser.add_mutually_exclusive_group()
    admin_group.add_argument("--admin", "-a", action="store_true", help="Grant administrator privileges")
    admin_group.add_argument("--no-admin", dest="no_admin", action="store_true", help="Remove administrator privileges")

    args = parser.parse_args()

    if args.root:
        os.environ["INVOKEAI_ROOT"] = args.root

    is_admin = None
    if args.admin:
        is_admin = True
    elif args.no_admin:
        is_admin = False

    if args.email:
        success = _modify_user_cli(args.email, args.name, args.password, is_admin)
    else:
        success = _modify_user_interactive()

    sys.exit(0 if success else 1)
