# InvokeAI Multi-User Administrator Guide

## Overview

This guide is for administrators managing a multi-user InvokeAI installation. It covers initial setup, user management, security best practices, and troubleshooting.

## Prerequisites

Before enabling multi-user support, ensure you have:

- InvokeAI installed and running
- Access to the server filesystem (for initial setup)
- Understanding of your deployment environment
- Backup of your existing data (recommended)

## Initial Setup

### Activating Multiuser Mode

To put InvokeAI into multiuser mode, you will need to add the option
`multiuser: true` to its configuration file. This file is located at
`INVOKEAI_ROOT/invokeai.yaml` With the InvokeAI backend halted, add
the new configuration option to the end of the file with a text editor
so that it looks like this:

```yaml
# Internal metadata - do not edit:
schema_version: 4.0.2

# Enable/disable multi-user mode
multiuser: true
```

Then restart the InvokeAI server backend from the command line or
using the launcher.

!!! note "Reverting to single-user mode"
	If at any time you wish to revert to single-user mode, simply comment
    out the `multiuser` line, or change "true" to "false". Then
	restart the server. Because of the way that browsers cache pages,
	users with open InvokeAI sessions may need to force-refresh their
	browsers.
	

### First Administrator Account

When InvokeAI starts for the first time in multi-user mode, you'll see the **Administrator Setup** dialog.

**Setup Steps:**

1. **Email Address**: Enter a valid email address (this becomes your username)

    * Example: `admin@example.com` or `admin@localhost` for testing
    * Must be a valid email format
    * Cannot be changed later without database access

2. **Display Name**: Enter a friendly name

    * Example: "System Administrator" or your real name
    * Can be changed later in your profile
    * Visible to other users in shared contexts

3. **Password**: Create a strong administrator password

    * **Minimum requirements:**

        * At least 8 characters long
        * Contains uppercase letters (A-Z)
        * Contains lowercase letters (a-z)
        * Contains numbers (0-9)

    * **Recommended:**

        * Use 12+ characters
        * Include special characters (!@#$%^&*)
        * Use a password manager to generate and store
        * Don't reuse passwords from other services

4. **Confirm Password**: Re-enter the password

5. Click **Create Administrator Account**

!!! warning "Important" 
    Store these credentials securely! The
    first administrator account can reset 
	the password to something new, but cannot 
	retrieve a lost one.

### Configuration

InvokeAI can run in single-user or multi-user mode, controlled by the `multiuser` configuration option in `invokeai.yaml`:

```yaml
# Enable/disable multi-user mode
multiuser: true   # Enable multi-user mode (requires authentication)
# multiuser: false  # Single-user mode (no authentication required)
# If the multiuser option is absent, single-user mode is used

# Database configuration
use_memory_db: false  # Use persistent database
db_path: databases/invokeai.db  # Database location

# Session configuration (multi-user mode only)
jwt_secret_key: "your-secret-key-here"  # Auto-generated if not specified
jwt_token_expiry_hours: 24  # Default session timeout
jwt_remember_me_days: 7  # "Remember me" duration
```

**Single-User Mode** (`multiuser: false` or option absent):
- No authentication required
- All functionality enabled by default
- All boards and images visible in unified view
- Ideal for personal use or trusted environments

**Multi-User Mode** (`multiuser: true`):
- Authentication required for access
- User isolation for boards, images, and workflows
- Role-based permissions enforced
- Ideal for shared servers or team environments

!!! warning "Mode Switching Behavior"
    **Switching to Single-User Mode:** If boards or images were created in multi-user mode, they will all be combined into a single unified view when switching to single-user mode.
    
    **Switching to Multi-User Mode:** Legacy boards and images created under single-user mode will be owned by an internal user named "system." Only the Administrator will have access to these legacy assets. A utility to migrate these legacy assets to another user will be part of a future release.

### Migration from Single-User

When upgrading from a single-user installation or switching modes:

1. **Automatic Migration**: The database will automatically migrate to multi-user schema when multi-user mode is first enabled
2. **Legacy Data Ownership**: Existing data (boards, images, workflows) created in single-user mode is assigned to an internal user named "system"
3. **Administrator Access**: Only administrators will have access to legacy "system"-owned assets when in multi-user mode
4. **No Data Loss**: All existing content is preserved

**Migration Process:**

```bash
# Backup your database first
cp databases/invokeai.db databases/invokeai.db.backup

# Enable multi-user mode in invokeai.yaml
# multiuser: true

# Start InvokeAI (migration happens automatically)
invokeai-web

# Complete the administrator setup dialog
# Legacy data will be owned by "system" user
```

!!! note "Legacy Asset Migration"
    A utility to migrate legacy "system"-owned assets to specific user accounts will be available in a future release. Until then, administrators can access and manage all legacy content.

## User Management

### Creating Users

**Via Web Interface (Coming Soon):**

!!! info "Web UI for User Management"
    A web-based user interface that allows administrators to manage users is coming in a future release. Until then, use the command-line scripts described below.

**Via Command Line Scripts:**

InvokeAI provides several command-line scripts in the `scripts/` directory for user management:

**useradd.py** - Add a new user:

```bash
# Interactive mode (prompts for details)
python scripts/useradd.py

# Create a regular user
python scripts/useradd.py \
  --email user@example.com \
  --password TempPass123 \
  --name "User Name"

# Create an administrator
python scripts/useradd.py \
  --email admin@example.com \
  --password AdminPass123 \
  --name "Admin Name" \
  --admin
```

**userlist.py** - List all users:

```bash
# List all users
python scripts/userlist.py

# Show detailed information
python scripts/userlist.py --verbose
```

**usermod.py** - Modify an existing user:

```bash
# Change display name
python scripts/usermod.py --email user@example.com --name "New Name"

# Promote to administrator
python scripts/usermod.py --email user@example.com --admin

# Demote from administrator
python scripts/usermod.py --email user@example.com --no-admin

# Deactivate account
python scripts/usermod.py --email user@example.com --deactivate

# Reactivate account
python scripts/usermod.py --email user@example.com --activate

# Change password
python scripts/usermod.py --email user@example.com --password NewPassword123
```

**userdel.py** - Delete a user:

```bash
# Delete a user (prompts for confirmation)
python scripts/userdel.py --email user@example.com

# Delete without confirmation
python scripts/userdel.py --email user@example.com --force
```

!!! tip "Script Usage"
    Run any script with `--help` to see all available options:
    ```bash
    python scripts/useradd.py --help
    ```

!!! warning "Command Line Management"
    - These scripts directly modify the database
    - Always backup your database before making changes
    - Changes take effect immediately (users may need to log in again)
    - Deleting a user permanently removes all their content

### Editing Users

**Via Command Line:**

Use `usermod.py` as described above to modify user properties.

!!! warning "Last Administrator"
    You cannot remove admin privileges from the last remaining administrator account.

### Resetting User Passwords

**Via Web Interface (Coming Soon):**

Web-based password reset functionality for administrators is coming in a future release.

**Via Command Line:**

```bash
# Reset a user's password
python scripts/usermod.py --email user@example.com --password NewTempPassword123
```

**Security Note:** Never send passwords via email or unsecured channels. Use secure communication methods.

### Deactivating Users

**Via Command Line:**

```bash
# Deactivate a user account
python scripts/usermod.py --email user@example.com --deactivate

# Reactivate a user account
python scripts/usermod.py --email user@example.com --activate
```

**Effects:**

- User cannot log in when deactivated
- Existing sessions are immediately invalidated
- User's data is preserved
- Can be reactivated at any time

### Deleting Users

**Via Command Line:**

```bash
# Delete a user (prompts for confirmation)
python scripts/userdel.py --email user@example.com

# Delete without confirmation prompt
python scripts/userdel.py --email user@example.com --force
```

**Important:**

- ⚠️ This action is **permanent**
- User's boards, images, and workflows are deleted
- Cannot be undone
- Consider deactivating instead of deleting

!!! warning "Data Loss"
    Deleting a user permanently removes all their content. Back up the database first if recovery might be needed.

### Viewing User Activity

**Queue Management:**

1. Navigate to **Admin** → **Queue Overview**
2. View all users' active and pending generations
3. Filter by user
4. Cancel stuck or problematic tasks

**User Statistics:**

- Number of boards created
- Number of images generated
- Storage usage (if enabled)
- Last login time

## Model Management

As an administrator, you have full access to model management.

### Adding Models

**Via Model Manager UI:**

1. Go to **Models** tab
2. Click **Add Model**
3. Choose installation method:
   - **From URL**: Provide HuggingFace repo or download URL
   - **From Local Path**: Scan local directories
   - **Import**: Import model from filesystem

**Supported Model Types:**

- Main models (Stable Diffusion, SDXL, FLUX)
- LoRA models
- ControlNet models
- VAE models
- Textual Inversions
- IP-Adapters

### Configuring Models

**Model Settings:**

- Display name
- Description
- Default generation settings (CFG, steps, scheduler)
- Variant selection (fp16/fp32)
- Model thumbnail image

**Default Settings:**

Set default parameters that users will start with:

1. Select a model
2. Go to **Default Settings** tab
3. Configure:
   - CFG Scale
   - Steps
   - Scheduler
   - VAE selection
4. Save settings

### Removing Models

1. Go to **Models** tab
2. Select model(s) to remove
3. Click **Delete**
4. Confirm deletion

!!! warning "Impact"
    Removing a model affects all users who may be using it in workflows or saved settings.

## Shared Boards

Shared boards enable collaboration between users while maintaining control.

!!! note "Future Feature"
	Board sharing will be implemented in a future release.

### Creating Shared Boards

1. Log in as administrator
2. Create a new board (or use existing board)
3. Right-click the board → **Share Board**
4. Add users and set permissions
5. Click **Save Sharing Settings**

### Permission Levels

| Level | View | Add Images | Edit/Delete | Manage Sharing |
|-------|------|------------|-------------|----------------|
| **Read** | ✅ | ❌ | ❌ | ❌ |
| **Write** | ✅ | ✅ | ✅ | ❌ |
| **Admin** | ✅ | ✅ | ✅ | ✅ |

**Permission Recommendations:**

- **Read**: For viewers who should see but not modify content
- **Write**: For active collaborators who add and organize images
- **Admin**: For trusted users who help manage the shared board

### Managing Shared Boards

**Add Users to Shared Board:**

1. Right-click shared board → **Manage Sharing**
2. Click **Add User**
3. Select user from dropdown
4. Choose permission level
5. Save changes

**Remove Users from Shared Board:**

1. Right-click shared board → **Manage Sharing**
2. Find user in list
3. Click **Remove**
4. Confirm removal

**Change User Permissions:**

1. Right-click shared board → **Manage Sharing**
2. Find user in list
3. Change permission dropdown
4. Save changes

### Shared Board Best Practices

- Give meaningful names to shared boards
- Document the board's purpose in the description
- Assign minimum necessary permissions
- Regularly audit access lists
- Remove users who no longer need access

## Security

### Password Policies

**Enforced Requirements:**

- Minimum 8 characters
- Must contain uppercase letters
- Must contain lowercase letters
- Must contain numbers

**Recommended Policies:**

- Require 12+ character passwords
- Include special characters
- Implement password rotation every 90 days
- Prevent password reuse
- Use multi-factor authentication (when available)

### Session Management

**Session Security and Token Management:**

This system uses stateless JWT tokens with HMAC signatures to
identify users after they provide their initial credentials. The
tokens will persist for 24 hours by default, or for 7 days if the user
clicks the "Remember me" checkbox at login. Expired tokens are
automatically rejected and the user will have to log in again.

At the client side, tokens are stored in browser localStorage. Logging
out clears them. No server-side session storage is required.

The tokens include the user's ID, email, and admin status, along with
an HMAC signature.

### Secret Key Management

**Important:** The JWT secret key must be kept confidential.

To generate tokens, each InvokeAI instance has a distinct secret JWT key that must be
kept confidential. The key is stored in the `app_settings` table of
the InvokeAI database with in a field value named `jwt_secret`.

The secret key is automatically generated during database creation or
migration. If you wish to change the key, you may generate a
replacement using either of these commands:


```bash
# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# OpenSSL
openssl rand -base64 32
```

Then cut and paste the printed secret into this Sqlite3 command:

```bash
sqlite3 INVOKE_ROOT/databases/invokeai.db 'update app_settings set value="THE_SECRET" where key="jwt_secret"'
```

(replace INVOKE_ROOT with your InvokeAI root directory and THE_SECRET
with the new secret).

After this, restart the server. All logged in users will be logged out
and will need to provide their usernames and passwords again.

### Hosting a Shared InvokeAI Instance

The multiuser feature allows you to run an InvokeAI backend that can
be accessed by your friends and family across your home network. It is
also possible to host a backend that is accessible over the Internet.

By default, InvokeAI runs on `localhost`, IP address `127.0.0.1`,
which is only accessible to browsers running on the same machine as
the backend. To make the backend accessible to any machine on your
home or work LAN, add the line `host: 0.0.0.0` to the InvokeAI
configuration file, usually stored at `INVOKE_ROOT/invokeai.yaml`.

Here is a minimal example.

```yaml
# Internal metadata - do not edit:
schema_version: 4.0.2

# Put user settings here - see https://invoke-ai.github.io/InvokeAI/configuration/:
multiuser: true
host: 0.0.0.0
```

After relaunching the backend you will be able to reach the server
from other machines on the LAN using the server machine's IP address
or hostname and port 9090.

#### Connecting to the Internet

!!! warning "Use at your own risk"
	The InvokeAI team has done its best to make the software free of
	exploitable bugs, but the software has not undergone a rigorous security
	audit or intrusion testing. Use at your own risk

It is also possible to create a (semi) public server accessible from
the Internet. The details of how to do this depend very much on your
home or corporate router/firewall system and are beyond the scope of
this document. 

If you expose InvokeAI to the Internet, there are a number of
precautions to take. Here is a brief list of recommended network
security practices.

**HTTPS Configuration:**

For internet deployments, always use HTTPS:

```yaml
# Use a reverse proxy like nginx or Traefik
# Example nginx configuration:

server {
    listen 443 ssl http2;
    server_name invoke.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:9090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

**Firewall Rules:**

It is best to restrict access to trusted networks and remote IP
addresses, or use a VPN to connect to your home network. Rate limit
connections to InvokeAI's authentication endpoint
`http://your.host:9090/login`.

**Backup and Recovery:**

It is a good idea to periodically backup your InvokeAI database,
images, and possibly models in the event of unauthorized use of a
publicly-accessible server.

**Manual Backup:**

```bash
# Stop InvokeAI
# Copy database file
cd INVOKE_ROOT
cp databases/invokeai.db databases/invokeai.db.$(date +%Y%m%d)

# Or create compressed backup
tar -czf invokeai_backup_$(date +%Y%m%d).tar.gz databases/
```

**Automated Backup Script:**

```bash
#!/bin/bash
# backup_invokeai.sh

INVOKE_ROOT="/path/to/invoke_root"
BACKUP_DIR="/path/to/backups"
DB_PATH="$INVOKE_ROOT/databases/invokeai.db"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Copy database
cp "$DB_PATH" "$BACKUP_DIR/invokeai_$DATE.db"

# Keep only last 30 days
find "$BACKUP_DIR" -name "invokeai_*.db" -mtime +30 -delete

echo "Backup completed: invokeai_$DATE.db"
```

**Schedule with cron:**

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /path/to/backup_invokeai.sh
```



```bash
# Stop InvokeAI
# Replace current database with backup
cd INVOKE_ROOT
cp databases/invokeai.db databases/invokeai.db.old  # Save current
cp databases/invokeai_backup.db databases/invokeai.db

# Restart InvokeAI
invokeai-web
```

**Disaster Recover - Complete System Backup:**

Include these directories/files:

- `databases/` - All database files
- `models/` - Installed models (if locally stored)
- `outputs/` - Generated images
- `invokeai.yaml` - Configuration file
- Any custom scripts or modifications

**Recovery Process:**

1. Install InvokeAI on new system
2. Restore configuration file
3. Restore database directory
4. Restore models and outputs
5. Verify file permissions
6. Start InvokeAI and test

## Troubleshooting

### User Cannot Login

**Symptom:** User reports unable to log in

**Diagnosis:**

1. Verify account exists and is active
   ```bash
   sqlite3 databases/invokeai.db "SELECT * FROM users WHERE email = 'user@example.com';"
   ```

2. Check password (have user try resetting)
3. Verify account is active (`is_active = 1`)
4. Check for account lockout (if implemented)

**Solutions:**

- Reset user password
- Reactivate disabled account
- Verify email address is correct
- Check system logs for auth errors

### Database Locked Errors

**Symptom:** "Database is locked" errors

**Causes:**

- Concurrent write operations
- Long-running transactions
- Backup process accessing database
- File system issues

**Solutions:**

```bash
# Check for locks
fuser databases/invokeai.db

# Increase timeout (in config)
# Or switch to WAL mode:
sqlite3 databases/invokeai.db "PRAGMA journal_mode=WAL;"
```

### Forgotten Admin Password

**Recovery Process:**

1. Stop InvokeAI
2. Direct database access:
   ```bash
   sqlite3 databases/invokeai.db
   ```

3. Reset admin password (requires password hash):
   ```sql
   -- Generate hash first using Python:
   -- from passlib.context import CryptContext
   -- pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
   -- print(pwd_context.hash("NewPassword123"))
   
   UPDATE users 
   SET password_hash = '$2b$12$...' 
   WHERE email = 'admin@example.com';
   ```

4. Restart InvokeAI

**Alternative:** Remove `jwt_secret_key` from config to trigger setup wizard (will create new admin).

### Performance Issues

**Symptom:** Slow generation or UI

**Diagnosis:**

1. Check active generation count
2. Review resource usage (CPU/GPU/RAM)
3. Check database size and performance
4. Review network latency

**Solutions:**

- Limit concurrent generations
- Increase hardware resources
- Optimize database (`VACUUM`, `ANALYZE`)
- Add indexes for slow queries
- Consider load balancing

### Migration Failures

**Symptom:** Database migration fails on upgrade

**Prevention:**

- Always backup before upgrading
- Test migration on copy of database
- Review migration logs

**Recovery:**

```bash
# Restore backup
cp databases/invokeai.db.backup databases/invokeai.db

# Try migration again with verbose logging
invokeai-web --log-level DEBUG
```

## Configuration Reference

### Complete Configuration Example for a Public Site

```yaml
# invokeai.yaml - Multi-user configuration

# Internal metadata - do not edit:
schema_version: 4.0.2

# Put user settings here
multiuser: true

# Server
host: "0.0.0.0"
port: 9090

# Performance
enable_partial_loading: true
precision: float16
pytorch_cuda_alloc_conf: "backend:cudaMallocAsync"
hashing_algorithm: blake3_multi
```
## Frequently Asked Questions

### How many users can InvokeAI support?

The backend will support dozens of concurrent users. However, because
the image generation queue is single-threaded, image generation tasks
are processed on a first-come, first-serve basis. This means that a
user may have to wait for all the other users' image generation jobs
to complete before their generation job starts to execute.

A future version of InvokeAI may support concurrent execution on
systems with multiple GPUs/graphics cards.

### Can I integrate with existing authentication systems?

OAuth2/OpenID Connect support is planned for a future release. Currently, InvokeAI uses its own authentication system.

### How do I audit user actions?

Full audit logging is planned for a future release. Currently, you can:

- Monitor the generation queue
- Review database changes
- Check application logs

### Can users have different model access?

Not in the current release. All users can view and use all installed models. Per-user model access is a possible enhancement.

### How do I handle user data when they leave?

Best practice:

1. Deactivate the account first
2. Transfer ownership of shared boards
3. After transition period, delete the account
4. Or keep the account deactivated for audit purposes

### What's the licensing impact of multi-user mode?

InvokeAI remains under its existing license. Multi-user mode does not change licensing terms.

## Getting Help

### Support Resources

- **Documentation**: [InvokeAI Docs](https://invoke-ai.github.io/InvokeAI/)
- **Discord**: [Join Community](https://discord.gg/ZmtBAhwWhy)
- **GitHub Issues**: [Report Problems](https://github.com/invoke-ai/InvokeAI/issues)
- **User Guide**: [For Users](user_guide.md)
- **API Guide**: [For Developers](api_guide.md)

### Reporting Issues

When reporting administrator issues, include:

- InvokeAI version
- Operating system and version
- Database size and user count
- Relevant log excerpts
- Steps to reproduce
- Expected vs actual behavior

## Additional Resources

- [User Guide](user_guide.md) - For end users
- [API Guide](api_guide.md) - For API consumers
- [Multiuser Specification](specification.md) - Technical details

---

**Need additional assistance?** Visit the [InvokeAI Discord](https://discord.gg/ZmtBAhwWhy) or file an issue on [GitHub](https://github.com/invoke-ai/InvokeAI/issues).
