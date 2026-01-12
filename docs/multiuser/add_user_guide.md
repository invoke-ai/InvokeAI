# User Management Utility

This directory contains the `add_user.py` script for managing InvokeAI users during development and testing.

## Quick Start

### Add a Regular User

```bash
python scripts/add_user.py --email testuser@test.local --password TestPass123 --name "Test User"
```

### Add an Admin User

```bash
python scripts/add_user.py --email admin@test.local --password AdminPass123 --name "Admin User" --admin
```

### Interactive Mode

Run without arguments to be prompted for details:

```bash
python scripts/add_user.py
```

## Password Requirements

Passwords must meet the following requirements:
- At least 8 characters long
- Contains at least one uppercase letter
- Contains at least one lowercase letter
- Contains at least one number

## Examples

```bash
# Add a regular user with display name
python scripts/add_user.py \
  --email alice@test.local \
  --password SecurePass123 \
  --name "Alice Johnson"

# Add an administrator
python scripts/add_user.py \
  --email admin@invokeai.local \
  --password AdminSecure456 \
  --name "System Administrator" \
  --admin

# Interactive mode (prompts for all details)
python scripts/add_user.py
```

## Testing Email Addresses

The script supports testing domains like `.local`, `.test`, and `.localhost` which are useful for development:

- `user@test.local`
- `admin@localhost`
- `testuser@invokeai.test`

## Troubleshooting

### "User with email already exists"

The email address is already in the database. Use a different email or remove the existing user first.

### "Password must be at least 8 characters long"

The password doesn't meet the minimum length requirement. Use a longer password.

### "Password must contain uppercase, lowercase, and numbers"

The password doesn't meet complexity requirements. Include at least:
- One uppercase letter (A-Z)
- One lowercase letter (a-z)  
- One digit (0-9)

## Database Location

The script uses the database path configured in your InvokeAI configuration. To find your database location:

```bash
python -c "from invokeai.app.services.config import get_config; print(get_config().db_path)"
```

## For Developers

The script can also be imported and used programmatically:

```python
from scripts.add_user import add_user_cli

# Add a user
success = add_user_cli(
    email="developer@test.local",
    password="DevPass123",
    display_name="Developer User",
    is_admin=False
)
```

## See Also

- Phase 6 Testing Guide: `docs/multiuser/phase6_testing.md`
- User Service Implementation: `invokeai/app/services/users/`
- Multiuser Specification: `docs/multiuser/specification.md`
