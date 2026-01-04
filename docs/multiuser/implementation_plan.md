# InvokeAI Multi-User Support - Implementation Plan

## 1. Overview

This document provides a detailed, step-by-step implementation plan for adding multi-user support to InvokeAI. It is designed to guide developers through the implementation process while maintaining code quality and minimizing disruption to existing functionality.

## 2. Implementation Approach

### 2.1 Principles
- **Minimal Changes**: Make surgical changes to existing code
- **Backward Compatibility**: Support existing single-user installations
- **Security First**: Implement security best practices from the start
- **Incremental Development**: Build and test in small, verifiable steps
- **Test Coverage**: Add tests for all new functionality

### 2.2 Development Strategy

1. Start with backend database and services
2. Build authentication layer
3. Update existing services for multi-tenancy
4. Develop frontend authentication
5. Update UI for multi-user features
6. Integration testing and security review

## 3. Prerequisites

### 3.1 Dependencies to Add

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "passlib[bcrypt]>=1.7.4",  # Password hashing
    "python-jose[cryptography]>=3.3.0",  # JWT tokens
    "python-multipart>=0.0.6",  # Form data parsing (already present)
    "email-validator>=2.0.0",  # Email validation
]
```

### 3.2 Development Environment Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests to ensure baseline
pytest tests/

# Start development server
python -m invokeai.app.run_app --dev_reload
```

## 4. Phase 1: Database Schema (Week 1)

### 4.1 Create Migration File

**File**: `invokeai/app/services/shared/sqlite_migrator/migrations/migration_25.py`

```python
import sqlite3
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

class Migration25Callback:
    """Migration to add multi-user support."""
    
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_users_table(cursor)
        self._create_user_sessions_table(cursor)
        self._create_user_invitations_table(cursor)
        self._create_shared_boards_table(cursor)
        self._update_boards_table(cursor)
        self._update_images_table(cursor)
        self._update_workflows_table(cursor)
        self._update_session_queue_table(cursor)
        self._update_style_presets_table(cursor)
        self._create_system_user(cursor)
    
    def _create_users_table(self, cursor: sqlite3.Cursor) -> None:
        """Create users table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
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
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);")
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS tg_users_updated_at
            AFTER UPDATE ON users FOR EACH ROW
            BEGIN
                UPDATE users SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE user_id = old.user_id;
            END;
        """)
    
    # ... implement other methods ...
    
    def _create_system_user(self, cursor: sqlite3.Cursor) -> None:
        """Create system user for backward compatibility."""
        cursor.execute("""
            INSERT OR IGNORE INTO users (user_id, email, display_name, password_hash, is_admin, is_active)
            VALUES ('system', 'system@invokeai.local', 'System', '', TRUE, TRUE);
        """)

def build_migration_25() -> Migration:
    """Build migration 25: Multi-user support."""
    return Migration(
        from_version=24,
        to_version=25,
        callback=Migration25Callback(),
    )
```

### 4.2 Update Migration Registry

**File**: `invokeai/app/services/shared/sqlite_migrator/migrations/__init__.py`

```python
from .migration_25 import build_migration_25

# Add to migrations list
def build_migrations() -> list[Migration]:
    return [
        # ... existing migrations ...
        build_migration_25(),
    ]
```

### 4.3 Testing
```bash
# Test migration
pytest tests/test_sqlite_migrator.py -v

# Manually test migration
python -m invokeai.app.run_app --use_memory_db
# Verify tables created
```

## 5. Phase 2: Authentication Service (Week 2)

### 5.1 Create Password Utilities

**File**: `invokeai/app/services/auth/password_utils.py`

```python
"""Password hashing and validation utilities."""
from passlib.context import CryptContext
from typing import Tuple

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate password meets requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    
    if not (has_upper and has_lower and has_digit):
        return False, "Password must contain uppercase, lowercase, and numbers"
    
    return True, ""
```

### 5.2 Create Token Service

**File**: `invokeai/app/services/auth/token_service.py`

```python
"""JWT token generation and validation."""
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional
from pydantic import BaseModel

SECRET_KEY = "your-secret-key-should-be-in-config"  # TODO: Move to config
ALGORITHM = "HS256"

class TokenData(BaseModel):
    user_id: str
    email: str
    is_admin: bool

def create_access_token(data: TokenData, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.model_dump()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(**payload)
    except JWTError:
        return None
```

### 5.3 Create User Service Base

**File**: `invokeai/app/services/users/users_base.py`

```python
"""Abstract base class for user service."""
from abc import ABC, abstractmethod
from typing import Optional
from .users_common import UserDTO, UserCreateRequest, UserUpdateRequest

class UserServiceABC(ABC):
    """High-level service for user management."""
    
    @abstractmethod
    def create(self, user_data: UserCreateRequest) -> UserDTO:
        """Create a new user."""
        pass
    
    @abstractmethod
    def get(self, user_id: str) -> Optional[UserDTO]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    def get_by_email(self, email: str) -> Optional[UserDTO]:
        """Get user by email."""
        pass
    
    @abstractmethod
    def update(self, user_id: str, changes: UserUpdateRequest) -> UserDTO:
        """Update user."""
        pass
    
    @abstractmethod
    def delete(self, user_id: str) -> None:
        """Delete user."""
        pass
    
    @abstractmethod
    def authenticate(self, email: str, password: str) -> Optional[UserDTO]:
        """Authenticate user credentials."""
        pass
```

### 5.4 Create User Service Implementation

**File**: `invokeai/app/services/users/users_default.py`

```python
"""Default implementation of user service."""
from uuid import uuid4
from .users_base import UserServiceABC
from .users_common import UserDTO, UserCreateRequest, UserUpdateRequest
from ..auth.password_utils import hash_password, verify_password
from ..shared.sqlite.sqlite_database import SqliteDatabase

class UserService(UserServiceABC):
    """SQLite-based user service."""
    
    def __init__(self, db: SqliteDatabase):
        self._db = db
    
    def create(self, user_data: UserCreateRequest) -> UserDTO:
        """Create a new user."""
        user_id = str(uuid4())
        password_hash = hash_password(user_data.password)
        
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO users (user_id, email, display_name, password_hash, is_admin)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, user_data.email, user_data.display_name, 
                 password_hash, user_data.is_admin)
            )
        
        return self.get(user_id)
    
    # ... implement other methods ...
```

### 5.5 Testing
```bash
# Create test file
# tests/app/services/users/test_user_service.py

pytest tests/app/services/users/ -v
```

## 6. Phase 3: Authentication Middleware (Week 3)

### 6.1 Create Auth Dependencies

**File**: `invokeai/app/api/auth_dependencies.py`

```python
"""FastAPI dependencies for authentication."""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated
from ..services.auth.token_service import verify_token, TokenData
from ..services.users.users_common import UserDTO

security = HTTPBearer()

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> TokenData:
    """Get current authenticated user from token."""
    token = credentials.credentials
    token_data = verify_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data

async def require_admin(
    current_user: Annotated[TokenData, Depends(get_current_user)]
) -> TokenData:
    """Require admin role."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Type aliases for route dependencies
CurrentUser = Annotated[TokenData, Depends(get_current_user)]
AdminUser = Annotated[TokenData, Depends(require_admin)]
```

### 6.2 Create Authentication Router

**File**: `invokeai/app/api/routers/auth.py`

```python
"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import timedelta
from ..auth_dependencies import CurrentUser
from ..dependencies import ApiDependencies
from ...services.auth.token_service import create_access_token, TokenData

auth_router = APIRouter(prefix="/v1/auth", tags=["authentication"])

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class LoginResponse(BaseModel):
    token: str
    user: dict
    expires_in: int

class SetupRequest(BaseModel):
    email: EmailStr
    display_name: str
    password: str

@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return token."""
    user_service = ApiDependencies.invoker.services.users
    user = user_service.authenticate(request.email, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Create token
    expires_delta = timedelta(days=7 if request.remember_me else 1)
    token_data = TokenData(
        user_id=user.user_id,
        email=user.email,
        is_admin=user.is_admin
    )
    token = create_access_token(token_data, expires_delta)
    
    return LoginResponse(
        token=token,
        user=user.model_dump(),
        expires_in=int(expires_delta.total_seconds())
    )

@auth_router.post("/logout")
async def logout(current_user: CurrentUser):
    """Logout current user."""
    # TODO: Implement token invalidation if using server-side sessions
    return {"success": True}

@auth_router.get("/me")
async def get_current_user_info(current_user: CurrentUser):
    """Get current user information."""
    user_service = ApiDependencies.invoker.services.users
    user = user_service.get(current_user.user_id)
    return user

@auth_router.post("/setup")
async def setup_admin(request: SetupRequest):
    """Set up initial administrator account."""
    user_service = ApiDependencies.invoker.services.users
    
    # Check if any admin exists
    # TODO: Implement count_admins method
    if user_service.has_admin():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Administrator already configured"
        )
    
    # Create admin user
    # TODO: Implement user creation with admin flag
    user = user_service.create_admin(request)
    
    return {"success": True, "user": user.model_dump()}
```

### 6.3 Register Auth Router

**File**: `invokeai/app/api_app.py` (modify)

```python
# Add import
from invokeai.app.api.routers import auth

# Add router registration (around line 135)
app.include_router(auth.auth_router, prefix="/api")
```

### 6.4 Testing
```bash
# Test authentication endpoints
pytest tests/app/routers/test_auth.py -v

# Manual testing with curl
curl -X POST http://localhost:9090/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@test.com","password":"test123"}'
```

## 7. Phase 4: Update Services for Multi-tenancy (Weeks 4-5)

### 7.1 Update Boards Service

**File**: `invokeai/app/services/boards/boards_default.py` (modify)

```python
# Add user_id parameter to methods
def create(self, board_name: str, user_id: str) -> BoardDTO:
    """Creates a board for a specific user."""
    # Add user_id to INSERT
    pass

def get_many(
    self,
    user_id: str,  # Add this parameter
    order_by: BoardRecordOrderBy,
    direction: SQLiteDirection,
    offset: int = 0,
    limit: int = 10,
    include_archived: bool = False,
) -> OffsetPaginatedResults[BoardDTO]:
    """Gets many boards for a specific user."""
    # Add WHERE user_id = ? OR is_shared = TRUE
    pass
```

**File**: `invokeai/app/api/routers/boards.py` (modify)

```python
from ..auth_dependencies import CurrentUser

@boards_router.get("/", response_model=OffsetPaginatedResults[BoardDTO])
async def list_boards(
    current_user: CurrentUser,  # Add this dependency
    # ... existing parameters ...
) -> OffsetPaginatedResults[BoardDTO]:
    """Gets a list of boards for the current user."""
    return ApiDependencies.invoker.services.boards.get_many(
        user_id=current_user.user_id,  # Add user filter
        # ... existing parameters ...
    )
```

### 7.2 Update Images Service

**File**: `invokeai/app/services/images/images_default.py` (modify)

Similar changes as boards - add user_id filtering to all queries.

### 7.3 Update Workflows Service

**File**: `invokeai/app/services/workflow_records/workflow_records_sqlite.py` (modify)

Add user_id and is_public filtering.

### 7.4 Update Session Queue Service

**File**: `invokeai/app/services/session_queue/session_queue_default.py` (modify)

Add user_id to queue items and filter by user unless admin.

### 7.5 Testing
```bash
# Test each updated service
pytest tests/app/services/boards/test_boards_multiuser.py -v
pytest tests/app/services/images/test_images_multiuser.py -v
pytest tests/app/services/workflows/test_workflows_multiuser.py -v
```

## 8. Phase 5: Frontend Authentication (Week 6)

### 8.1 Create Auth Slice

**File**: `invokeai/frontend/web/src/features/auth/store/authSlice.ts`

```typescript
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface User {
  user_id: string;
  email: string;
  display_name: string;
  is_admin: boolean;
}

interface AuthState {
  isAuthenticated: boolean;
  token: string | null;
  user: User | null;
  isLoading: boolean;
}

const initialState: AuthState = {
  isAuthenticated: false,
  token: localStorage.getItem('auth_token'),
  user: null,
  isLoading: false,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setCredentials: (state, action: PayloadAction<{ token: string; user: User }>) => {
      state.token = action.payload.token;
      state.user = action.payload.user;
      state.isAuthenticated = true;
      localStorage.setItem('auth_token', action.payload.token);
    },
    logout: (state) => {
      state.token = null;
      state.user = null;
      state.isAuthenticated = false;
      localStorage.removeItem('auth_token');
    },
  },
});

export const { setCredentials, logout } = authSlice.actions;
export default authSlice.reducer;
```

### 8.2 Create Login Page Component

**File**: `invokeai/frontend/web/src/features/auth/components/LoginPage.tsx`

```typescript
import { useState } from 'react';
import { useLoginMutation } from '../api/authApi';
import { useAppDispatch } from '@/app/store';
import { setCredentials } from '../store/authSlice';

export const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [login, { isLoading, error }] = useLoginMutation();
  const dispatch = useAppDispatch();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const result = await login({ email, password, remember_me: rememberMe }).unwrap();
      dispatch(setCredentials({ token: result.token, user: result.user }));
    } catch (err) {
      // Error handled by RTK Query
    }
  };

  return (
    <div className="login-container">
      <form onSubmit={handleSubmit}>
        <h1>Sign In to InvokeAI</h1>
        
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          required
        />
        
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          required
        />
        
        <label>
          <input
            type="checkbox"
            checked={rememberMe}
            onChange={(e) => setRememberMe(e.target.checked)}
          />
          Remember me
        </label>
        
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Signing in...' : 'Sign In'}
        </button>
        
        {error && <div className="error">Login failed. Please check your credentials.</div>}
      </form>
    </div>
  );
};
```

### 8.3 Create Protected Route Wrapper

**File**: `invokeai/frontend/web/src/features/auth/components/ProtectedRoute.tsx`

```typescript
import { Navigate } from 'react-router-dom';
import { useAppSelector } from '@/app/store';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requireAdmin?: boolean;
}

export const ProtectedRoute = ({ children, requireAdmin = false }: ProtectedRouteProps) => {
  const { isAuthenticated, user } = useAppSelector((state) => state.auth);

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (requireAdmin && !user?.is_admin) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};
```

### 8.4 Update API Configuration

**File**: `invokeai/frontend/web/src/services/api/index.ts` (modify)

```typescript
// Add auth header to all requests
import { createApi } from '@reduxjs/toolkit/query/react';

const baseQuery = fetchBaseQuery({
  baseUrl: '/api',
  prepareHeaders: (headers, { getState }) => {
    const token = (getState() as RootState).auth.token;
    if (token) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    return headers;
  },
});
```

## 9. Phase 6: Frontend UI Updates (Week 7)

### 9.1 Update App Root

**File**: `invokeai/frontend/web/src/main.tsx` (modify)

```typescript
import { LoginPage } from './features/auth/components/LoginPage';
import { ProtectedRoute } from './features/auth/components/ProtectedRoute';

// Wrap main app in ProtectedRoute
<Router>
  <Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route path="/*" element={
      <ProtectedRoute>
        <App />
      </ProtectedRoute>
    } />
  </Routes>
</Router>
```

### 9.2 Add User Menu

**File**: `invokeai/frontend/web/src/features/ui/components/UserMenu.tsx`

```typescript
import { useAppSelector, useAppDispatch } from '@/app/store';
import { logout } from '@/features/auth/store/authSlice';
import { useNavigate } from 'react-router-dom';

export const UserMenu = () => {
  const user = useAppSelector((state) => state.auth.user);
  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  const handleLogout = () => {
    dispatch(logout());
    navigate('/login');
  };

  return (
    <div className="user-menu">
      <span>{user?.display_name || user?.email}</span>
      {user?.is_admin && <span className="admin-badge">Admin</span>}
      <button onClick={handleLogout}>Logout</button>
    </div>
  );
};
```

### 9.3 Hide Model Manager for Non-Admin

**File**: `invokeai/frontend/web/src/features/modelManager/ModelManager.tsx` (modify)

```typescript
import { useAppSelector } from '@/app/store';

export const ModelManager = () => {
  const user = useAppSelector((state) => state.auth.user);

  if (!user?.is_admin) {
    return (
      <div className="access-denied">
        <h2>Model Management</h2>
        <p>This feature is only available to administrators.</p>
      </div>
    );
  }

  // ... existing model manager code ...
};
```

## 10. Phase 7: Testing & Security (Weeks 8-9)

### 10.1 Unit Tests

Create comprehensive tests for:
- Password hashing and validation
- Token generation and verification
- User service methods
- Authorization checks
- Data isolation queries

### 10.2 Integration Tests

Test complete flows:
- User registration and login
- Password reset
- Multi-user data isolation
- Shared board access
- Admin operations

### 10.3 Security Testing

- SQL injection tests
- XSS prevention tests
- CSRF protection
- Authorization bypass attempts
- Session hijacking prevention

### 10.4 Performance Testing

- Authentication overhead
- Query performance with user filters
- Concurrent user sessions

## 11. Phase 8: Documentation (Week 10)

### 11.1 User Documentation
- Getting started guide
- Login and account management
- Using shared boards
- Understanding permissions

### 11.2 Administrator Documentation
- Setup guide
- User management
- Security best practices
- Backup and restore

### 11.3 API Documentation
- Update OpenAPI schema
- Add authentication examples
- Document new endpoints

## 12. Phase 9: Migration Support (Week 11)

### 12.1 Migration Wizard

Create CLI tool to assist with migration:

```bash
python -m invokeai.app.migrate_to_multiuser
```

Features:
- Detect existing installation
- Prompt for admin credentials
- Migrate existing data
- Validate migration
- Rollback on error

### 12.2 Backward Compatibility

Add config option to disable auth:

```yaml
# invokeai.yaml
auth_enabled: false  # Legacy single-user mode
```

## 13. Rollout Strategy

### 13.1 Beta Testing

1. Internal testing with core team (1 week)
2. Closed beta with selected users (2 weeks)
3. Open beta announcement (2 weeks)
4. Stable release

### 13.2 Communication Plan
- Blog post announcing feature
- Documentation updates
- Migration guide
- FAQ and troubleshooting
- Discord announcement

### 13.3 Support Plan
- Monitor Discord for issues
- Create GitHub issues template for auth bugs
- Provide migration assistance
- Collect feedback for improvements

## 14. Success Criteria

- [ ] All unit tests pass (>90% coverage for new code)
- [ ] All integration tests pass
- [ ] Security review completed with no critical findings
- [ ] Performance benchmarks met (no more than 10% overhead)
- [ ] Documentation complete and reviewed
- [ ] Beta testing completed successfully
- [ ] Migration from single-user tested and verified
- [ ] Zero data loss incidents
- [ ] Positive feedback from beta users

## 15. Risk Mitigation

### 15.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| Database migration failures | Extensive testing, backup requirements, rollback procedures |
| Performance degradation | Index optimization, query profiling, load testing |
| Security vulnerabilities | Security review, penetration testing, CodeQL scans |
| Authentication bugs | Comprehensive testing, beta period, gradual rollout |

### 15.2 User Experience Risks

| Risk | Mitigation |
|------|------------|
| Migration confusion | Clear documentation, migration wizard, support channels |
| Login friction | Long session timeout, remember me option, clear messaging |
| Feature discoverability | Updated UI, tooltips, onboarding flow |

## 16. Maintenance Plan

### 16.1 Ongoing Support
- Monitor error logs for auth failures
- Regular security updates
- Password policy reviews
- Session management optimization

### 16.2 Future Enhancements
- OAuth2/OpenID Connect
- Two-factor authentication
- Advanced permission system
- Team/group management
- Audit logging

## 17. Conclusion

This implementation plan provides a structured approach to adding multi-user support to InvokeAI. The phased approach allows for:

1. **Incremental Development**: Build and test in small steps
2. **Early Validation**: Test core functionality early
3. **Risk Mitigation**: Identify issues before they become problems
4. **Quality Assurance**: Comprehensive testing at each phase
5. **User Focus**: Beta testing and feedback incorporation

By following this plan, the development team can deliver a robust, secure, and user-friendly multi-user system while maintaining the quality and reliability that InvokeAI users expect.

## 18. Quick Reference

### Key Files to Create
- `migration_25.py` - Database migration
- `password_utils.py` - Password hashing
- `token_service.py` - JWT token management
- `users_base.py` - User service interface
- `users_default.py` - User service implementation
- `auth_dependencies.py` - FastAPI auth dependencies
- `routers/auth.py` - Authentication endpoints
- `authSlice.ts` - Frontend auth state
- `LoginPage.tsx` - Login UI component
- `ProtectedRoute.tsx` - Route protection

### Key Files to Modify
- `api_app.py` - Register auth router
- `config_default.py` - Add auth config options
- `boards_default.py` - Add user filtering
- `images_default.py` - Add user filtering
- `main.tsx` - Add route protection
- All existing routers - Add auth dependencies

### Commands
```bash
# Run tests
pytest tests/ -v

# Run specific test suite
pytest tests/app/services/users/ -v

# Run with coverage
pytest tests/ --cov=invokeai.app.services --cov-report=html

# Run development server
python -m invokeai.app.run_app --dev_reload

# Run database migration
python -m invokeai.app.migrate

# Create new migration
python -m invokeai.app.create_migration "Add multi-user support"
```

### Useful Links
- [FastAPI Security Docs](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT.io](https://jwt.io/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [SQLite Foreign Keys](https://www.sqlite.org/foreignkeys.html)
