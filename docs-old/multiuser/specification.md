# InvokeAI Multi-User Support - Detailed Specification

## 1. Executive Summary

This document provides a comprehensive specification for adding multi-user support to InvokeAI. The feature will enable a single InvokeAI instance to support multiple isolated users, each with their own generation settings, image boards, and workflows, while maintaining administrative controls for model management and system configuration.

## 2. Overview

### 2.1 Goals
- Enable multiple users to share a single InvokeAI instance
- Provide user isolation for personal content (boards, images, workflows, settings)
- Maintain centralized model management by administrators
- Support shared boards for collaboration
- Provide secure authentication and authorization
- Minimize impact on existing single-user installations

### 2.2 Non-Goals
- Real-time collaboration features (multiple users editing same workflow simultaneously)
- Advanced team management features (in initial release)
- Migration of existing multi-user enterprise edition data
- Support for external identity providers (in initial release, can be added later)

## 3. User Roles and Permissions

### 3.1 Administrator Role
**Capabilities:**

- Full access to all InvokeAI features
- Model management (add, delete, configure models)
- User management (create, edit, delete users)
- View and manage all users' queue sessions
- Access system configuration
- Create and manage shared boards
- Grant/revoke administrative privileges to other users

**Restrictions:**

- Cannot delete their own account if they are the last administrator
- Cannot revoke their own admin privileges if they are the last administrator

### 3.2 Regular User Role
**Capabilities:**

- Create, edit, and delete their own image boards
- Upload and manage their own assets
- Use all image generation tools (linear, canvas, upscale, workflow tabs)
- Create, edit, save, and load workflows
- Access public/shared workflows
- View and manage their own queue sessions
- Adjust personal UI preferences (theme, hotkeys, etc.)
- Access shared boards (read/write based on permissions)
- **View model configurations** (read-only access to model manager)
- **View model details, default settings, and metadata**

**Restrictions:**

- Cannot add, delete, or edit models
- **Can view but cannot modify model manager settings** (read-only access)
- Cannot reidentify, convert, or update model paths
- Cannot upload or change model thumbnail images
- Cannot save changes to model default settings
- Cannot perform bulk delete operations on models
- Cannot view or modify other users' boards, images, or workflows
- Cannot cancel or modify other users' queue sessions
- Cannot access system configuration
- Cannot manage users or permissions

### 3.3 Future Role Considerations
- **Viewer Role**: Read-only access (future enhancement)
- **Team/Group-based Permissions**: Organizational hierarchy (future enhancement)

## 4. Authentication System

### 4.1 Authentication Method
- **Primary Method**: Username and password authentication with secure password hashing
- **Password Hashing**: Use bcrypt or Argon2 for password storage
- **Session Management**: JWT tokens or secure session cookies
- **Token Expiration**: Configurable session timeout (default: 7 days for "remember me", 24 hours otherwise)

### 4.2 Initial Administrator Setup
**First-time Launch Flow:**

1. Application detects no administrator account exists
2. Displays mandatory setup dialog (cannot be skipped)
3. Prompts for:
   - Administrator username (email format recommended)
   - Administrator display name
   - Strong password (minimum requirements enforced)
   - Password confirmation
4. Stores hashed credentials in configuration
5. Creates administrator account in database
6. Proceeds to normal login screen

**Reset Capability:**

- Administrators can be reset by manually editing the config file
- Requires access to server filesystem (intentional security measure)
- Database maintains user records; config file contains root admin credentials

### 4.3 Password Requirements
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character (optional but recommended)
- Not in common password list

### 4.4 Login Flow

1. User navigates to InvokeAI URL
2. If not authenticated, redirect to login page
3. User enters username/email and password
4. Optional "Remember me" checkbox for extended session
5. Backend validates credentials
6. On success: Generate session token, redirect to application
7. On failure: Display error, allow retry with rate limiting (prevent brute force)

### 4.5 Logout Flow
- User clicks logout button
- Frontend clears session token
- Backend invalidates session (if using server-side sessions)
- Redirect to login page

### 4.6 Future Authentication Enhancements
- OAuth2/OpenID Connect support
- Two-factor authentication (2FA)
- SSO integration
- API key authentication for programmatic access

## 5. User Management

### 5.1 User Creation (Administrator)
**Flow:**

1. Administrator navigates to user management interface
2. Clicks "Add User" button
3. Enters user information:
   - Email address (required, used as username)
   - Display name (optional, defaults to email)
   - Role (User or Administrator)
   - Initial password or "Send invitation email"
4. System validates email uniqueness
5. System creates user account
6. If invitation mode:
   - Generate one-time secure token
   - Send email with setup link
   - Link expires after 7 days
7. If direct password mode:
   - Administrator provides initial password
   - User must change on first login

**Invitation Email Flow:**

1. User receives email with unique link
2. Link contains secure token
3. User clicks link, redirected to setup page
4. User enters desired password
5. Token validated and consumed (single-use)
6. Account activated
7. User redirected to login page

### 5.2 User Profile Management
**User Self-Service:**

- Update display name
- Change password (requires current password)
- Update email address (requires verification)
- Manage UI preferences
- View account creation date and last login

**Administrator Actions:**

- Edit user information (name, email)
- Reset user password (generates reset link)
- Toggle administrator privileges
- Assign to groups (future feature)
- Suspend/unsuspend account
- Delete account (with data retention options)

### 5.3 Password Reset Flow
**User-Initiated (Future Enhancement):**

1. User clicks "Forgot Password" on login page
2. Enters email address
3. System sends password reset link (if email exists)
4. User clicks link, enters new password
5. Password updated, user can login

**Administrator-Initiated:**

1. Administrator selects user
2. Clicks "Send Password Reset"
3. System generates reset token and link
4. Email sent to user
5. User follows same flow as user-initiated reset

## 6. Data Model and Database Schema

### 6.1 New Tables

#### 6.1.1 users
```sql
CREATE TABLE users (
    user_id TEXT NOT NULL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    last_login_at DATETIME
);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_admin ON users(is_admin);
CREATE INDEX idx_users_is_active ON users(is_active);
```

#### 6.1.2 user_sessions
```sql
CREATE TABLE user_sessions (
    session_id TEXT NOT NULL PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    expires_at DATETIME NOT NULL,
    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    last_activity_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    user_agent TEXT,
    ip_address TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_user_sessions_token_hash ON user_sessions(token_hash);
```

#### 6.1.3 user_invitations
```sql
CREATE TABLE user_invitations (
    invitation_id TEXT NOT NULL PRIMARY KEY,
    email TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    invited_by_user_id TEXT NOT NULL,
    expires_at DATETIME NOT NULL,
    used_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    FOREIGN KEY (invited_by_user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX idx_user_invitations_email ON user_invitations(email);
CREATE INDEX idx_user_invitations_token_hash ON user_invitations(token_hash);
CREATE INDEX idx_user_invitations_expires_at ON user_invitations(expires_at);
```

#### 6.1.4 shared_boards
```sql
CREATE TABLE shared_boards (
    board_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    permission TEXT NOT NULL CHECK(permission IN ('read', 'write', 'admin')),
    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    PRIMARY KEY (board_id, user_id),
    FOREIGN KEY (board_id) REFERENCES boards(board_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX idx_shared_boards_user_id ON shared_boards(user_id);
CREATE INDEX idx_shared_boards_board_id ON shared_boards(board_id);
```

### 6.2 Modified Tables

#### 6.2.1 boards
```sql
-- Add columns:
ALTER TABLE boards ADD COLUMN user_id TEXT NOT NULL DEFAULT 'system';
ALTER TABLE boards ADD COLUMN is_shared BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE boards ADD COLUMN created_by_user_id TEXT;

-- Add foreign key (requires recreation in SQLite):
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
FOREIGN KEY (created_by_user_id) REFERENCES users(user_id) ON DELETE SET NULL

-- Add indices:
CREATE INDEX idx_boards_user_id ON boards(user_id);
CREATE INDEX idx_boards_is_shared ON boards(is_shared);
```

#### 6.2.2 images
```sql
-- Add column:
ALTER TABLE images ADD COLUMN user_id TEXT NOT NULL DEFAULT 'system';

-- Add foreign key:
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE

-- Add index:
CREATE INDEX idx_images_user_id ON images(user_id);
```

#### 6.2.3 workflows
```sql
-- Add columns:
ALTER TABLE workflows ADD COLUMN user_id TEXT NOT NULL DEFAULT 'system';
ALTER TABLE workflows ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT FALSE;

-- Add foreign key:
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE

-- Add indices:
CREATE INDEX idx_workflows_user_id ON workflows(user_id);
CREATE INDEX idx_workflows_is_public ON workflows(is_public);
```

#### 6.2.4 session_queue
```sql
-- Add column:
ALTER TABLE session_queue ADD COLUMN user_id TEXT NOT NULL DEFAULT 'system';

-- Add foreign key:
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE

-- Add index:
CREATE INDEX idx_session_queue_user_id ON session_queue(user_id);
```

#### 6.2.5 style_presets
```sql
-- Add columns:
ALTER TABLE style_presets ADD COLUMN user_id TEXT NOT NULL DEFAULT 'system';
ALTER TABLE style_presets ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT FALSE;

-- Add foreign key:
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE

-- Add indices:
CREATE INDEX idx_style_presets_user_id ON style_presets(user_id);
CREATE INDEX idx_style_presets_is_public ON style_presets(is_public);
```

### 6.3 Migration Strategy

1. Create new user tables (users, user_sessions, user_invitations, shared_boards)
2. Create default 'system' user for backward compatibility
3. Update existing data to reference 'system' user
4. Add foreign key constraints
5. Version as database migration (e.g., migration_25.py)

### 6.4 Migration for Existing Installations
- Single-user installations: Prompt to create admin account on first launch after update
- Existing data migration: Administrator can specify an arbitrary user account to hold legacy data (can be the admin account or a separate user)
- System provides UI during migration to choose destination user for existing data

## 7. API Endpoints

### 7.1 Authentication Endpoints

#### POST /api/v1/auth/setup
- Initialize first administrator account
- Only works if no admin exists
- Body: `{ email, display_name, password }`
- Response: `{ success, user }`

#### POST /api/v1/auth/login
- Authenticate user
- Body: `{ email, password, remember_me? }`
- Response: `{ token, user, expires_at }`

#### POST /api/v1/auth/logout
- Invalidate current session
- Headers: `Authorization: Bearer <token>`
- Response: `{ success }`

#### GET /api/v1/auth/me
- Get current user information
- Headers: `Authorization: Bearer <token>`
- Response: `{ user }`

#### POST /api/v1/auth/change-password
- Change current user's password
- Body: `{ current_password, new_password }`
- Headers: `Authorization: Bearer <token>`
- Response: `{ success }`

### 7.2 User Management Endpoints (Admin Only)

#### GET /api/v1/users
- List all users (paginated)
- Query params: `offset`, `limit`, `search`, `role_filter`
- Response: `{ users[], total, offset, limit }`

#### POST /api/v1/users
- Create new user
- Body: `{ email, display_name, is_admin, send_invitation?, initial_password? }`
- Response: `{ user, invitation_link? }`

#### GET /api/v1/users/{user_id}
- Get user details
- Response: `{ user }`

#### PATCH /api/v1/users/{user_id}
- Update user
- Body: `{ display_name?, is_admin?, is_active? }`
- Response: `{ user }`

#### DELETE /api/v1/users/{user_id}
- Delete user
- Query params: `delete_data` (true/false)
- Response: `{ success }`

#### POST /api/v1/users/{user_id}/reset-password
- Send password reset email
- Response: `{ success, reset_link }`

### 7.3 Shared Boards Endpoints

#### POST /api/v1/boards/{board_id}/share
- Share board with users
- Body: `{ user_ids[], permission: 'read' | 'write' | 'admin' }`
- Response: `{ success, shared_with[] }`

#### GET /api/v1/boards/{board_id}/shares
- Get board sharing information
- Response: `{ shares[] }`

#### DELETE /api/v1/boards/{board_id}/share/{user_id}
- Remove board sharing
- Response: `{ success }`

### 7.4 Modified Endpoints

All existing endpoints will be modified to:

1. Require authentication (except setup/login)
2. Filter data by current user (unless admin viewing all)
3. Enforce permissions (e.g., model management requires admin)
4. Include user context in operations

Example modifications:
- `GET /api/v1/boards` → Returns only user's boards + shared boards
- `POST /api/v1/session/queue` → Associates queue item with current user
- `GET /api/v1/queue` → Returns all items for admin, only user's items for regular users

## 8. Frontend Changes

### 8.1 New Components

#### LoginPage
- Email/password form
- "Remember me" checkbox
- Login button
- Forgot password link (future)
- Branding and welcome message

#### AdministratorSetup
- Modal dialog (cannot be dismissed)
- Administrator account creation form
- Password strength indicator
- Terms/welcome message

#### UserManagementPage (Admin only)
- User list table
- Add user button
- User actions (edit, delete, reset password)
- Search and filter
- Role toggle

#### UserProfilePage
- Display user information
- Change password form
- UI preferences
- Account details

#### BoardSharingDialog
- User picker/search
- Permission selector
- Share button
- Current shares list

### 8.2 Modified Components

#### App Root
- Add authentication check
- Redirect to login if not authenticated
- Handle session expiration
- Add global error boundary for auth errors

#### Navigation/Header
- Add user menu with logout
- Display current user name
- Admin indicator badge

#### ModelManagerTab
- Hide/disable for non-admin users
- Show "Admin only" message

#### QueuePanel
- Filter by current user (for non-admin)
- Show all with user indicators (for admin)
- Disable actions on other users' items (for non-admin)

#### BoardsPanel
- Show personal boards section
- Show shared boards section
- Add sharing controls to board actions

### 8.3 State Management

New Redux slices/zustand stores:
- `authSlice`: Current user, authentication status, token
- `usersSlice`: User list for admin interface
- `sharingSlice`: Board sharing state

Updated slices:
- `boardsSlice`: Include shared boards, ownership info
- `queueSlice`: Include user filtering
- `workflowsSlice`: Include public/private status

## 9. Configuration

### 9.1 New Config Options

Add to `InvokeAIAppConfig`:

```python
# Authentication
auth_enabled: bool = True  # Enable/disable multi-user auth
session_expiry_hours: int = 24  # Default session expiration
session_expiry_hours_remember: int = 168  # "Remember me" expiration (7 days)
password_min_length: int = 8  # Minimum password length
require_strong_passwords: bool = True  # Enforce password complexity

# Session tracking
enable_server_side_sessions: bool = False  # Optional server-side session tracking

# Audit logging
audit_log_auth_events: bool = True  # Log authentication events
audit_log_admin_actions: bool = True  # Log administrative actions

# Email (optional - for invitations and password reset)
email_enabled: bool = False
smtp_host: str = ""
smtp_port: int = 587
smtp_username: str = ""
smtp_password: str = ""
smtp_from_address: str = ""
smtp_from_name: str = "InvokeAI"

# Initial admin (stored as hash)
admin_email: Optional[str] = None
admin_password_hash: Optional[str] = None
```

### 9.2 Backward Compatibility

- If `auth_enabled = False`, system runs in legacy single-user mode
- All data belongs to implicit "system" user
- No authentication required
- Smooth upgrade path for existing installations

## 10. Security Considerations

### 10.1 Password Security
- Never store passwords in plain text
- Use bcrypt or Argon2id for password hashing
- Implement proper salt generation
- Enforce password complexity requirements
- Implement rate limiting on login attempts
- Consider password breach checking (Have I Been Pwned API)

### 10.2 Session Security
- Use cryptographically secure random tokens
- Implement token rotation
- Set appropriate cookie flags (HttpOnly, Secure, SameSite)
- Implement session timeout and renewal
- Invalidate sessions on logout
- Clean up expired sessions periodically

### 10.3 Authorization
- Always verify user identity from session token (never trust client)
- Check permissions on every API call
- Implement principle of least privilege
- Validate user ownership of resources before operations
- Implement proper error messages (avoid information leakage)

### 10.4 Data Isolation
- Strict separation of user data in database queries
- Prevent SQL injection via parameterized queries
- Validate all user inputs
- Implement proper access control checks
- Audit trail for sensitive operations

### 10.5 API Security
- Implement rate limiting on sensitive endpoints
- Use HTTPS in production (enforce via config)
- Implement CSRF protection
- Validate and sanitize all inputs
- Implement proper CORS configuration
- Add security headers (CSP, X-Frame-Options, etc.)

### 10.6 Deployment Security
- Document secure deployment practices
- Recommend reverse proxy configuration (nginx, Apache)
- Provide example configurations for HTTPS
- Document firewall requirements
- Recommend network isolation strategies

## 11. Email Integration (Optional)

**Note**: Email/SMTP configuration is optional. Many administrators will not have ready access to an outgoing SMTP server. When email is not configured, the system provides fallback mechanisms by displaying setup links directly in the admin UI.

### 11.1 Email Templates

#### User Invitation
```
Subject: You've been invited to InvokeAI

Hello,

You've been invited to join InvokeAI by [Administrator Name].

Click the link below to set up your account:
[Setup Link]

This link expires in 7 days.

---
InvokeAI
```

#### Password Reset
```
Subject: Reset your InvokeAI password

Hello [User Name],

A password reset was requested for your account.

Click the link below to reset your password:
[Reset Link]

This link expires in 24 hours.

If you didn't request this, please ignore this email.

---
InvokeAI
```

### 11.2 Email Service
- Support SMTP configuration
- Use secure connection (TLS)
- Handle email failures gracefully
- Implement email queue for reliability
- Log email activities (without sensitive data)
- Provide fallback for no-email deployments (show links in admin UI)

## 12. Testing Requirements

### 12.1 Unit Tests
- Authentication service (password hashing, validation)
- Authorization checks
- Token generation and validation
- User management operations
- Shared board permissions
- Data isolation queries

### 12.2 Integration Tests
- Complete authentication flows
- User creation and invitation
- Password reset flow
- Multi-user data isolation
- Shared board access
- Session management
- Admin operations

### 12.3 Security Tests
- SQL injection prevention
- XSS prevention
- CSRF protection
- Session hijacking prevention
- Brute force protection
- Authorization bypass attempts

### 12.4 Performance Tests
- Authentication overhead
- Query performance with user filters
- Concurrent user sessions
- Database scalability with many users

## 13. Documentation Requirements

### 13.1 User Documentation
- Getting started with multi-user InvokeAI
- Login and account management
- Using shared boards
- Understanding permissions
- Troubleshooting authentication issues

### 13.2 Administrator Documentation
- Setting up multi-user InvokeAI
- User management guide
- Creating and managing shared boards
- Email configuration
- Security best practices
- Backup and restore with user data

### 13.3 Developer Documentation
- Authentication architecture
- API authentication requirements
- Adding new multi-user features
- Database schema changes
- Testing multi-user features

### 13.4 Migration Documentation
- Upgrading from single-user to multi-user
- Data migration strategies
- Rollback procedures
- Common issues and solutions

## 14. Future Enhancements

### 14.1 Phase 2 Features
- **OAuth2/OpenID Connect integration** (deferred from initial release to keep scope manageable)
- Two-factor authentication
- API keys for programmatic access
- Enhanced team/group management
- Advanced permission system (roles and capabilities)

### 14.2 Phase 3 Features
- SSO integration (SAML, LDAP)
- User quotas and limits
- Resource usage tracking
- Advanced collaboration features
- Workflow template library with permissions
- Model access controls per user/group

## 15. Success Metrics

### 15.1 Functionality Metrics
- Successful user authentication rate
- Zero unauthorized data access incidents
- All tests passing (unit, integration, security)
- API response time within acceptable limits

### 15.2 Usability Metrics
- User setup completion time < 2 minutes
- Login time < 2 seconds
- Clear error messages for all auth failures
- Positive user feedback on multi-user features

### 15.3 Security Metrics
- No critical security vulnerabilities identified
- CodeQL scan passes
- Penetration testing completed
- Security best practices followed

## 16. Risks and Mitigations

### 16.1 Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Performance degradation with user filtering | Medium | Low | Index optimization, query caching |
| Database migration failures | High | Low | Thorough testing, rollback procedures |
| Session management complexity | Medium | Medium | Use proven libraries (PyJWT), extensive testing |
| Auth bypass vulnerabilities | High | Low | Security review, penetration testing |

### 16.2 UX Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Confusion in migration for existing users | Medium | High | Clear documentation, migration wizard |
| Friction from additional login step | Low | High | Remember me option, long session timeout |
| Complexity of admin interface | Medium | Medium | Intuitive UI design, user testing |

### 16.3 Operational Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Email delivery failures | Low | Medium | Show links in UI, document manual methods |
| Lost admin password | High | Low | Document recovery procedure, config reset |
| User data conflicts in migration | Medium | Low | Data validation, backup requirements |

## 17. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Database schema design and migration
- Basic authentication service
- Password hashing and validation
- Session management

### Phase 2: Backend API (Weeks 3-4)
- Authentication endpoints
- User management endpoints
- Authorization middleware
- Update existing endpoints with auth

### Phase 3: Frontend Auth (Weeks 5-6)
- Login page and flow
- Administrator setup
- Session management
- Auth state management

### Phase 4: Multi-tenancy (Weeks 7-9)
- User isolation in all services
- Shared boards implementation
- Queue permission filtering
- Workflow public/private

### Phase 5: Admin Interface (Weeks 10-11)
- User management UI
- Board sharing UI
- Admin-specific features
- User profile page

### Phase 6: Testing & Polish (Weeks 12-13)
- Comprehensive testing
- Security audit
- Performance optimization
- Documentation
- Bug fixes

### Phase 7: Beta & Release (Week 14+)
- Beta testing with selected users
- Feedback incorporation
- Final testing
- Release preparation
- Documentation finalization

## 18. Acceptance Criteria

- [ ] Administrator can set up initial account on first launch
- [ ] Users can log in with email and password
- [ ] Users can change their password
- [ ] Administrators can create, edit, and delete users
- [ ] User data is properly isolated (boards, images, workflows)
- [ ] Shared boards work correctly with permissions
- [ ] Non-admin users cannot access model management
- [ ] Queue filtering works correctly for users and admins
- [ ] Session management works correctly (expiry, renewal, logout)
- [ ] All security tests pass
- [ ] API documentation is updated
- [ ] User and admin documentation is complete
- [ ] Migration from single-user works smoothly
- [ ] Performance is acceptable with multiple concurrent users
- [ ] Backward compatibility mode works (auth disabled)

## 19. Design Decisions

The following design decisions have been approved for implementation:

1. **OAuth2 Priority**: OAuth2/OpenID Connect integration will be a **future enhancement**. The initial release will focus on username/password authentication to keep scope manageable.

2. **Email Requirement**: Email/SMTP configuration is **optional**. Many administrators will not have ready access to an outgoing SMTP server. The system will provide fallback mechanisms (showing setup links directly in the admin UI) when email is not configured.

3. **Data Migration**: During migration from single-user to multi-user mode, the administrator will be given the **option to specify an arbitrary user account** to hold legacy data. The admin account can be used for this purpose if the administrator wishes.

4. **API Compatibility**: Authentication will be **required on all APIs**, but authentication will not be required if multi-user support is disabled (backward compatibility mode with `auth_enabled: false`).

5. **Session Storage**: The system will use **JWT tokens with optional server-side session tracking**. This provides scalability while allowing administrators to enable server-side tracking if needed.

6. **Audit Logging**: The system will **log authentication events and admin actions**. This provides accountability and security monitoring for critical operations.

## 20. Conclusion

This specification provides a comprehensive blueprint for implementing multi-user support in InvokeAI. The design prioritizes:

- **Security**: Proper authentication, authorization, and data isolation
- **Usability**: Intuitive UI, smooth migration, minimal friction
- **Scalability**: Efficient database design, performant queries
- **Maintainability**: Clean architecture, comprehensive testing
- **Flexibility**: Future enhancement paths, optional features

The phased implementation approach allows for iterative development and testing, while the detailed specifications ensure all stakeholders have clear expectations of the final system.
