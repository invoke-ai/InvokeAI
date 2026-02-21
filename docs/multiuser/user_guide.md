# InvokeAI Multi-User Guide

## Overview

InvokeAI supports both single-user and multi-user modes. In
single-user mode, no login is required and you have access to all
features. In multi-user mode, multiple people can use the same
InvokeAI instance while keeping their work private and organized.

### Single-User vs Multi-User Mode

**Single-User Mode:**

- No login required - direct access to InvokeAI
- All functionality enabled by default
- All boards and images visible in a unified view
- Ideal for personal use or trusted environments
- Enabled when `multiuser: false` in config or option is absent

**Multi-User Mode:**

- Secure login required for access
- User isolation for boards, images, and workflows
- Role-based permissions (Administrator vs Regular User)
- Ideal for shared servers or team environments
- Enabled when `multiuser: true` in config

!!! note "Mode Switching"
    
	If you switch from multi-user mode to single-user mode, 
	all boards and images from different users will be combined 
	into a single unified view. When switching back to multi-user
	mode, they will be separated again by user ownership.

## Getting Started

### Initial Setup (First Time in Multi-User Mode)

If you're the first person to access a fresh InvokeAI installation in multi-user mode, you'll see the **Administrator Setup** dialog:

1. Enter your email address (this will be your username)
2. Create a display name
3. Choose a strong password that meets the requirements:
   - At least 8 characters long
   - Contains uppercase letters
   - Contains lowercase letters
   - Contains numbers
4. Confirm your password
5. Click **Create Administrator Account**

You'll now be taken to a login screen and can enter the credentials
you just created.

### Accessing InvokeAI

**In Single-User Mode:**

1. Navigate to your InvokeAI URL (e.g., `http://localhost:9090`)
2. You'll go directly to the InvokeAI interface
3. No login required - start creating immediately!

**In Multi-User Mode:**

1. Navigate to your InvokeAI URL (e.g., `http://localhost:9090`)
2. You'll see the login screen
3. Enter your email address and password provided by your administrator
4. Click **Sign In**

!!! tip "Remember Me"
    In multi-user mode, check the "Remember me" box to stay logged in for 7 days. Otherwise, your session will expire after 24 hours.

## Understanding User Roles (Multi-User Mode Only)

In single-user mode, you have access to all features without restrictions. In multi-user mode, InvokeAI has two user roles:

### Regular User

As a regular user, you can:

- ✅ Create and manage your own image boards
- ✅ Generate images using all AI tools (Linear, Canvas, Upscale, Workflows)
- ✅ Create, save, and load your own workflows
- ✅ Access workflows marked as public
- ✅ View your own generation queue
- ✅ Customize your UI preferences (theme, hotkeys, etc.)
- ✅ Access shared boards (based on permissions granted to you) (FUTURE FEATURE)
- ✅ **View available models** (read-only access to Model Manager)

You cannot:

- ❌ Add, delete, or modify models
- ❌ View or modify other users' boards, images, or workflows
- ❌ Manage user accounts
- ❌ Access system configuration
- ❌ View or cancel other users' generation tasks

!!! tip "The generation queue"

	When two or more users are accessing InvokeAI at the same time,
	their image generation jobs will be placed on the session queue on
	a first-come, first-serve basis. This means that you will have to
	wait for other users' image rendering jobs to complete before
	yours will start.
	
	When another user's job is running, you will see the image
	generation progress bar and a queue badge that reads `X/Y`, where
	"X" is the number of jobs you have queued and "Y" is the total
	number of jobs queued, including your own and others.
	
	You can also pull up the Queue tab in order to see where your job
	is in relationship to other queued tasks.

### Administrator

Administrators have all regular user capabilities, plus:

- ✅ Full model management (add, delete, configure models)
- ✅ Create and manage user accounts
- ✅ View and manage all users' generation queues
- ✅ Create and manage shared boards (FUTURE FEATURE)
- ✅ Access system configuration
- ✅ Grant or revoke admin privileges

## Working with Your Content

### Image Boards

Image boards help organize your generated images. Each user has their own private boards.

**Creating a Board:**

1. Click the **+** button in the Boards panel
2. Enter a board name
3. Press Enter or click Create

**Managing Boards:**

- Click a board to select it
- Generated images will automatically be added to the selected board
- Right-click a board for options (rename, delete, archive)
- Drag images between boards to reorganize

**Board Visibility:**

- Your boards are private by default
- Only administrators can create shared boards (FUTURE FEATURE)
- You'll see shared boards you have access to in a separate section

### Workflows

Workflows are reusable generation templates that you create in the Workflow Editor.

**Creating a Workflow:**

1. Go to the **Workflows** tab
2. Build your workflow using nodes
3. Click **Save** and give it a name
4. Your workflow is saved to your personal library

**Workflow Privacy:**

- Your workflows are private by default
- Only you can see and edit your workflows
- Administrators can mark workflows as "public" for all users to access
- Public workflows appear in everyone's workflow library but remain read-only

### Your Generation Queue

The queue shows your pending and running generation tasks.

**Queue Features:**

- View your current and completed generations
- Cancel pending tasks
- Re-run previous generations
- Monitor progress in real-time

**Queue Isolation:**

- You will see your own queue items, as well as the items generated by
  either users, but the generation parameters (e.g. prompts) for other
  users' are hidden for privacy reasons.
- Administrators can view all queues for troubleshooting
- Your generations won't interfere with other users' tasks

## Using Shared Boards (FUTURE FEATURE)

Shared boards are a feature that will be added in a future
release. Administrators will able to designate certain boards as being
accessible to multiple users, allowing for collaboration among users
while maintaining security.

### Accessing Shared Boards

Shared boards appear in your Boards panel marked with a sharing icon. You can:

- View images on shared boards (if you have read access)
- Add images to shared boards (if you have write access)
- Use shared boards like your personal boards

### Permission Levels

Shared boards have three permission levels:

| Permission | View Images | Add Images | Edit/Delete | Manage Sharing |
|------------|-------------|------------|-------------|----------------|
| **Read** | ✅ | ❌ | ❌ | ❌ |
| **Write** | ✅ | ✅ | ✅ | ❌ |
| **Admin** | ✅ | ✅ | ✅ | ✅ |

!!! note "Shared boards"
    Only administrators will be able to create shared boards and
    assign initial permissions.

## Viewing Models (Read-Only)

Regular users have read-only access to the Model Manager, allowing you to:

**What You Can View:**

- ✅ Browse all available models
- ✅ See model details and configurations
- ✅ View default settings for each model
- ✅ Check model metadata and descriptions
- ✅ See which models are installed

**What You Cannot Do:**

- ❌ Install new models
- ❌ Delete or modify existing models
- ❌ Change model configurations
- ❌ Upload or change model images
- ❌ Convert models between formats

**Accessing the Model Manager:**

1. Click on the **Models** tab in the navigation
2. Browse available models
3. Click on any model to view its details

!!! tip "Need a New Model?"
    If you need a model that isn't installed, ask your administrator to add it.

## Customizing Your Experience

### Personal Preferences

Your UI preferences are saved to your account:

- **Theme**: Choose between light and dark modes
- **Hotkeys**: Customize keyboard shortcuts
- **Canvas Settings**: Default zoom, grid visibility, etc.
- **Generation Defaults**: Default values for width, height, steps, etc.

These settings are stored per-user and won't affect other users.

### Profile Settings (Multi-User Mode)

In multi-user mode, access your profile by clicking your name in the top-right corner:

**Display Name:** Update how your name appears throughout the UI

**Change Password:**

!!! info "Password Changes"
    A web-based interface for users to change their own passwords is coming in a future release. Until then, contact your administrator to reset your password if needed.

## Security Best Practices

### Password Security

- Use a strong, unique password
- Don't share your password with others
- Change your password regularly
- Use a password manager to store complex passwords

### Session Security

- Log out when using a shared computer
- Be aware of your session timeout (24 hours or 7 days with "remember me")
- Your session will automatically expire for security
- You'll need to log in again after the session expires

### Data Privacy

- Your boards, images, and workflows are private by default
- Other users cannot access your content unless explicitly shared
- Only administrators can see all users' content for management purposes

## Troubleshooting

### Cannot Log In

**Issue:** Login fails with "Incorrect email or password"

**Solutions:**

- Verify you're entering the correct email address
- Check that Caps Lock is off
- Try typing the password slowly to avoid mistakes
- Contact your administrator if you've forgotten your password

**Issue:** Login fails with "Account is disabled"

**Solution:** Contact your administrator to reactivate your account

### Session Expired

**Issue:** You're suddenly logged out and see "Session expired"

**Explanation:** Sessions expire after 24 hours (or 7 days with "remember me")

**Solution:** Simply log in again with your credentials

### Cannot Access Features

**Issue:** Features like Model Manager show "Admin privileges required"

**Explanation:** Some features are restricted to administrators

**Solution:** 

- For model viewing: You can view but not modify models
- For user management: Contact an administrator
- For system configuration: Contact an administrator

### Missing Boards or Images

**Issue:** Boards or images you created are not visible

**Possible Causes:**

1. **Filter Applied:** Check if a filter is hiding content
2. **Wrong User:** Ensure you're logged in with the correct account
3. **Archived Board:** Check the "Show Archived" option

**Solution:** 

- Clear any active filters
- Verify you're logged in as the right user
- Check archived items

### Slow Performance

**Issue:** Generation or UI feels slower than expected

**Possible Causes:**

- Other users generating images simultaneously
- Server resource limits
- Network latency

**Solutions:**

- Check the queue to see if others are generating
- Wait for current generations to complete
- Contact administrator if persistent

### Generation Stuck in Queue

**Issue:** Your generation is queued but not starting

**Possible Causes:**

- Server is processing other users' generations
- Server resources are fully utilized
- Technical issue with the server

**Solutions:**

- Wait for your turn in the queue
- Check if your generation is paused
- Contact administrator if stuck for extended period

## Common Tasks

### Changing Your Password

!!! note This is a FUTURE FEATURE. For now, the Administrator must change/reset a user's password using command-line tools.

1. Click your display name (top-right corner)
2. Select **Change Password**
3. Enter current password
4. Enter new password (8+ characters, mixed case, numbers)
5. Confirm new password
6. Click **Update Password**

### Creating a New Board

1. Navigate to the Gallery or Canvas tab
2. Find the Boards panel (usually on the left)
3. Click the **+ New Board** button
4. Type a descriptive name
5. Press Enter

### Saving a Workflow

1. Create or edit a workflow in the Workflows tab
2. Click **Save** in the top bar
3. Enter a workflow name
4. Optionally add a description
5. Click **Save Workflow**

### Finding a Public Workflow

!!! note Sharing of workflows is a FUTURE FEATURE, not yet implemented

1. Go to the **Workflows** tab
2. Open the workflow library
3. Public workflows are marked with a 🌐 icon
4. Click to load and use the workflow

### Logging Out

1. Click your display name (top-right corner)
2. Select **Logout**
3. You'll be redirected to the login screen

## Frequently Asked Questions

### Can other users see my images?

No, unless you add them to a shared board (FUTURE FEATURE). All your personal boards and images are private.

### Can I share my workflows with others?

Not directly. Ask your administrator to mark workflows as public if you want to share them.

### How long do sessions last?

- 24 hours by default
- 7 days if you check "Remember me" during login

### Can I use the API with multi-user mode?

Yes, but you'll need to authenticate with a JWT token. See the [API Guide](api_guide.md) for details.

### What happens if I forget my password?

Contact your administrator. They can reset your password for you.

### Can I have multiple sessions?

Yes, you can log in from multiple devices or browsers simultaneously. All sessions will use the same account and see the same content.

### Why can't I see the Model Manager "Add Models" tab?

Regular users can see the Models tab but with read-only access. Check that you're logged in and try refreshing the page.

### How do I know if I'm an administrator?

Administrators see an "Admin" badge next to their name in the top-right corner and have access to additional features like User Management.

### Can I request admin privileges?

Yes, ask your current administrator to grant you admin
privileges. Admin privileges will give you the ability to see all
other user's boards and images, as well as to add models and change
various server-wide settings.

## Getting Help

### Support Channels

- **Administrator:** Contact your system administrator for account issues
- **Documentation:** Check the [FAQ](../faq.md) for common issues
- **Community:** Join the [Discord](https://discord.gg/ZmtBAhwWhy) for help
- **Bug Reports:** File issues on [GitHub](https://github.com/invoke-ai/InvokeAI/issues)

### Reporting Issues

When reporting an issue, include:

- Your role (regular user or administrator)
- What you were trying to do
- What happened instead
- Any error messages you saw
- Your browser and operating system

## Additional Resources

- [Administrator Guide](admin_guide.md) - For administrators managing users and the system
- [API Guide](api_guide.md) - For developers using the InvokeAI API
- [Multiuser Specification](specification.md) - Technical details about the feature
- [InvokeAI Documentation](../index.md) - Main documentation hub

---

**Need more help?** Contact your administrator or visit the [InvokeAI Discord](https://discord.gg/ZmtBAhwWhy).
