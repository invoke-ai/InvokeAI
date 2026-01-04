# Multi-User Support for InvokeAI

This directory contains the detailed specification and implementation plan for adding multi-user support to InvokeAI.

## 📄 Documents

### 1. [Detailed Specification](./multiuser_specification.md)
Comprehensive technical specification covering:
- User roles and permissions
- Authentication system design
- Database schema changes
- API endpoint specifications
- Frontend component requirements
- Security considerations
- Email integration (optional)
- Testing requirements
- Documentation requirements
- Future enhancements
- Risk assessment
- Success criteria

### 2. [Implementation Plan](./multiuser_implementation_plan.md)
Step-by-step development guide covering:
- Phase-by-phase implementation timeline
- Code examples for each component
- File-by-file changes required
- Testing strategy
- Migration approach
- Rollout strategy
- Maintenance plan
- Quick reference guide

## 🎯 Quick Overview

### What This Feature Adds

**For Regular Users:**
- Secure login with email/password
- Personal image boards and workflows
- Isolated generation queue
- Custom UI preferences
- Access to shared collaborative boards

**For Administrators:**
- Full system management capabilities
- User account management
- Model management (add/remove/configure)
- Create and manage shared boards
- View and manage all user queues
- System configuration access

### Key Features

✅ **Secure Authentication**
- Password hashing with bcrypt/Argon2
- JWT token-based sessions
- Configurable session timeouts
- Rate limiting on login attempts

✅ **Data Isolation**
- Each user has separate boards, images, and workflows
- Database-level enforcement of data ownership
- Shared boards with granular permissions

✅ **Role-Based Access Control**
- Administrator role with full access
- Regular user role with restricted access
- Future support for custom roles

✅ **Backward Compatibility**
- Optional authentication (can be disabled)
- Smooth migration from single-user installations
- Minimal impact on existing deployments

## 📊 Implementation Status

### Phase Status
- [x] Phase 1: Specification & Documentation ✅
- [ ] Phase 2: Database Schema Design
- [ ] Phase 3: Backend - Authentication Service
- [ ] Phase 4: Backend - Multi-tenancy Updates
- [ ] Phase 5: Backend - API Updates
- [ ] Phase 6: Frontend - Authentication UI
- [ ] Phase 7: Frontend - UI Updates
- [ ] Phase 8: Testing & Documentation
- [ ] Phase 9: Security Review

**Current Status**: Specification Complete - Ready for Review

## 🚀 Getting Started (For Developers)

### Prerequisites
```bash
# Install dependencies
pip install -e ".[dev]"

# Additional dependencies for multi-user support
pip install passlib[bcrypt] python-jose[cryptography] email-validator
```

### Development Workflow

1. **Review Specification**
   - Read [multiuser_specification.md](./multiuser_specification.md)
   - Understand the requirements and architecture

2. **Follow Implementation Plan**
   - Reference [multiuser_implementation_plan.md](./multiuser_implementation_plan.md)
   - Implement phase by phase
   - Test each phase thoroughly

3. **Testing**
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Run with coverage
   pytest tests/ --cov=invokeai.app --cov-report=html
   ```

4. **Local Development**
   ```bash
   # Start with in-memory database for testing
   python -m invokeai.app.run_app --use_memory_db --dev_reload
   ```

## 📋 Technical Architecture

### Backend Components

```
invokeai/app/
├── services/
│   ├── auth/                    # Authentication utilities
│   │   ├── password_utils.py   # Password hashing
│   │   └── token_service.py    # JWT token management
│   ├── users/                   # User management service
│   │   ├── users_base.py       # Abstract interface
│   │   ├── users_default.py    # SQLite implementation
│   │   └── users_common.py     # DTOs and types
│   └── shared/
│       └── sqlite_migrator/
│           └── migrations/
│               └── migration_25.py  # Multi-user schema
├── api/
│   ├── auth_dependencies.py    # FastAPI auth dependencies
│   └── routers/
│       └── auth.py             # Authentication endpoints
```

### Frontend Components

```
frontend/web/src/
├── features/
│   ├── auth/
│   │   ├── store/
│   │   │   └── authSlice.ts        # Auth state management
│   │   ├── components/
│   │   │   ├── LoginPage.tsx       # Login UI
│   │   │   ├── ProtectedRoute.tsx  # Route protection
│   │   │   └── UserMenu.tsx        # User menu component
│   │   └── api/
│   │       └── authApi.ts          # Auth API endpoints
```

### Database Schema

```
users                    # User accounts
├── user_id (PK)
├── email (UNIQUE)
├── password_hash
├── is_admin
└── is_active

user_sessions            # Active sessions
├── session_id (PK)
├── user_id (FK)
├── token_hash
└── expires_at

boards                   # Modified for multi-user
├── board_id (PK)
├── user_id (FK)         # NEW: Owner
├── is_shared            # NEW: Sharing flag
└── ...

shared_boards           # NEW: Board sharing
├── board_id (FK)
├── user_id (FK)
└── permission
```

## 🔒 Security Considerations

### Critical Security Features

1. **Password Security**
   - Bcrypt hashing with appropriate work factor
   - No plain-text password storage
   - Password strength validation

2. **Session Management**
   - Secure JWT token generation
   - Token expiration and refresh
   - Server-side session tracking (optional)

3. **Authorization**
   - Role-based access control
   - Database-level data isolation
   - API endpoint protection

4. **Input Validation**
   - Email validation
   - SQL injection prevention
   - XSS prevention

### Security Testing Requirements

- [ ] SQL injection testing
- [ ] XSS vulnerability testing
- [ ] CSRF protection verification
- [ ] Authorization bypass testing
- [ ] Session hijacking prevention
- [ ] CodeQL security scan
- [ ] Penetration testing (recommended)

## 📖 Documentation

### For Users
- Getting Started Guide (to be created)
- Login and Account Management (to be created)
- Understanding Roles and Permissions (to be created)
- Using Shared Boards (to be created)

### For Administrators
- Administrator Setup Guide (to be created)
- User Management Guide (to be created)
- Security Best Practices (to be created)
- Backup and Recovery (to be created)

### For Developers
- [Detailed Specification](./multiuser_specification.md) ✅
- [Implementation Plan](./multiuser_implementation_plan.md) ✅
- API Documentation (to be generated)
- Testing Guide (to be created)

## 🎯 Timeline

### Estimated Timeline: 14 weeks

- **Weeks 1-2**: Database schema and migration
- **Weeks 3-4**: Backend authentication service
- **Weeks 5-6**: Frontend authentication UI
- **Weeks 7-9**: Multi-tenancy updates
- **Weeks 10-11**: Admin interface and features
- **Weeks 12-13**: Testing and polish
- **Week 14+**: Beta testing and release

## 🤝 Contributing

### How to Contribute

1. **Review Phase**
   - Review the specification document
   - Provide feedback on the design
   - Suggest improvements or alternatives

2. **Implementation Phase**
   - Pick a phase from the implementation plan
   - Follow the coding standards
   - Write tests for your code
   - Submit PR with documentation

3. **Testing Phase**
   - Test beta releases
   - Report bugs and issues
   - Suggest UX improvements

### Code Review Checklist

- [ ] Follows implementation plan
- [ ] Includes unit tests
- [ ] Includes integration tests (if applicable)
- [ ] Updates documentation
- [ ] No security vulnerabilities
- [ ] Backward compatible (or migration provided)
- [ ] Performance acceptable
- [ ] Code follows project style guide

## ❓ FAQ

### Q: Will this break my existing installation?
A: No. The feature includes a migration path and can be disabled for single-user mode.

### Q: Is OAuth2/OpenID Connect supported?
A: Not in the initial release, but it's planned for a future enhancement.

### Q: Can I run this in production?
A: After the initial release and security review, yes. Follow the security best practices in the documentation.

### Q: How do I reset the administrator password?
A: Edit the config file to remove the admin credentials, then restart the application to trigger the setup flow again.

### Q: Can users collaborate in real-time?
A: Not in the initial release. Shared boards allow asynchronous collaboration.

### Q: Will this affect performance?
A: Minimal impact expected (<10% overhead). Performance testing will verify this.

## 📞 Support

### Getting Help

- **Development Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues (use "multi-user" label)
- **Security Issues**: security@invoke.ai (do not file public issues)
- **General Support**: Discord #support channel

### Reporting Issues

When reporting issues, include:
- InvokeAI version
- Operating system
- Authentication enabled/disabled
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs (remove sensitive data)

## 📜 License

This feature is part of InvokeAI and is licensed under the same terms as the main project.

## 🙏 Acknowledgments

This feature addresses requirements from the community and replaces functionality that was previously available in the enterprise edition. Thanks to all community members who provided feedback and requirements.

---

**Status**: Specification Complete - Awaiting Review
**Last Updated**: January 4, 2026
**Next Steps**: Review and feedback on specification, begin Phase 2 implementation
