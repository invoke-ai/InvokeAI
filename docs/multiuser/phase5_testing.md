# Phase 5: Frontend Authentication Testing Guide

## Overview

This document provides comprehensive testing instructions for Phase 5 of the multiuser implementation - Frontend Authentication.

**Status**: ✅ COMPLETE  
**Implementation Date**: January 10, 2026  
**Implementation Branch**: `copilot/implement-phase-5-multiuser`

---

## Components Implemented

### 1. Redux State Management
- **Auth Slice** (`features/auth/store/authSlice.ts`)
  - Manages authentication state (token, user, loading status)
  - Persists token to localStorage
  - Provides selectors for authentication status

### 2. API Endpoints
- **Auth API** (`services/api/endpoints/auth.ts`)
  - `POST /api/v1/auth/login` - User authentication
  - `POST /api/v1/auth/logout` - User logout
  - `GET /api/v1/auth/me` - Get current user info
  - `POST /api/v1/auth/setup` - Initial administrator setup

### 3. UI Components
- **LoginPage** - User authentication interface
- **AdministratorSetup** - Initial admin account creation
- **ProtectedRoute** - Route wrapper for authentication checking

### 4. Routing
- Integrated react-router-dom
- `/login` - Login page
- `/setup` - Administrator setup
- `/*` - Protected application routes

---

## Prerequisites

### Backend Setup
Ensure Phases 1-4 are complete and the backend is running:
1. Backend must have migration 25 applied (users table exists)
2. Auth endpoints must be available at `/api/v1/auth/*`
3. Backend should be running on `localhost:9090` (default)

### Frontend Setup
```bash
cd invokeai/frontend/web
pnpm install
pnpm build
```

---

## Manual Testing Scenarios

### Scenario 1: Initial Setup Flow

**Objective**: Verify administrator account creation on first launch.

**Steps**:
1. Ensure no admin exists in database (fresh install or reset database)
2. Navigate to `http://localhost:5173/` (dev mode) or `http://localhost:9090/` (production)
3. Application should redirect to `/setup`
4. Fill in the administrator setup form:
   - Email: `admin@test.com`
   - Display Name: `Test Administrator`
   - Password: `TestPassword123` (meets complexity requirements)
   - Confirm Password: `TestPassword123`
5. Click "Create Administrator Account"

**Expected Results**:
- Form validates password strength (8+ chars, uppercase, lowercase, numbers)
- Passwords must match
- On success, redirects to `/login`
- Admin account is created in database

**Verification**:
```bash
# Check database for admin user
sqlite3 invokeai.db "SELECT user_id, email, display_name, is_admin FROM users WHERE email='admin@test.com';"
```

---

### Scenario 2: Login Flow

**Objective**: Verify user can authenticate successfully.

**Steps**:
1. Navigate to `http://localhost:5173/login` (or get redirected from main app)
2. Enter credentials:
   - Email: `admin@test.com`
   - Password: `TestPassword123`
3. Optional: Check "Remember me for 7 days"
4. Click "Sign In"

**Expected Results**:
- Successful login redirects to main application (`/`)
- Token is stored in localStorage (key: `auth_token`)
- Redux state is updated with user information
- Authorization header is added to subsequent API requests

**Verification**:
```javascript
// In browser console:
localStorage.getItem('auth_token')  // Should return JWT token
```

---

### Scenario 3: Protected Routes

**Objective**: Verify unauthenticated users cannot access the main application.

**Steps**:
1. Clear localStorage: `localStorage.clear()`
2. Navigate to `http://localhost:5173/`

**Expected Results**:
- Application redirects to `/login`
- Main application content is not displayed
- User cannot bypass authentication

---

### Scenario 4: Token Persistence

**Objective**: Verify token persists across browser sessions.

**Steps**:
1. Login with "Remember me" checked
2. Close browser tab
3. Open new tab and navigate to application
4. Check if user is still authenticated

**Expected Results**:
- User remains logged in
- No redirect to login page
- Application loads normally

---

### Scenario 5: Logout Flow

**Objective**: Verify user can logout successfully.

**Steps**:
1. Login to application
2. Click logout button (to be implemented in Phase 6)
3. OR manually call logout: `dispatch(logout())` in browser console

**Expected Results**:
- Token is removed from localStorage
- Redux state is cleared
- User is redirected to `/login`
- Cannot access protected routes without re-authenticating

**Verification**:
```javascript
// In browser console:
localStorage.getItem('auth_token')  // Should return null
```

---

### Scenario 6: Invalid Credentials

**Objective**: Verify proper error handling for invalid credentials.

**Steps**:
1. Navigate to login page
2. Enter invalid credentials:
   - Email: `admin@test.com`
   - Password: `WrongPassword`
3. Click "Sign In"

**Expected Results**:
- Error message displayed: "Login failed. Please check your credentials."
- User remains on login page
- No token stored
- No navigation occurs

---

### Scenario 7: Weak Password Validation (Setup)

**Objective**: Verify password strength requirements are enforced.

**Steps**:
1. Navigate to `/setup`
2. Try various weak passwords:
   - `short` - Too short
   - `alllowercase123` - No uppercase
   - `ALLUPPERCASE123` - No lowercase
   - `NoNumbers` - No digits

**Expected Results**:
- Form validation prevents submission
- Appropriate error message displayed for each case
- "Create Administrator Account" button disabled when password is invalid

---

### Scenario 8: API Authorization Headers

**Objective**: Verify Authorization header is added to API requests.

**Steps**:
1. Login successfully
2. Open browser DevTools → Network tab
3. Perform any action that makes an API call (e.g., list boards)
4. Inspect request headers

**Expected Results**:
- All API requests (except `/auth/login` and `/auth/setup`) include:
  ```
  Authorization: Bearer <token>
  ```
- Token matches value in localStorage

---

## Automated Testing

### Running Tests

```bash
cd invokeai/frontend/web

# Run all frontend tests
pnpm test:no-watch

# Run with UI
pnpm test:ui

# Run with coverage
pnpm test:no-watch --coverage
```

**Note**: Automated tests for Phase 5 components should be added in follow-up work. Current focus is on integration and manual testing.

---

## Integration with Backend

### Test with Running Backend

1. Start backend server:
```bash
# From repository root
python -m invokeai.app.run_app
```

2. Start frontend dev server:
```bash
cd invokeai/frontend/web
pnpm dev
```

3. Navigate to `http://localhost:5173/`
4. Follow manual testing scenarios above

### API Endpoint Testing

Use cURL or Postman to test endpoints directly:

```bash
# Setup admin
curl -X POST http://localhost:9090/api/v1/auth/setup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@test.com",
    "display_name": "Administrator",
    "password": "TestPassword123"
  }'

# Login
curl -X POST http://localhost:9090/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@test.com",
    "password": "TestPassword123",
    "remember_me": true
  }'

# Get current user (requires token)
curl -X GET http://localhost:9090/api/v1/auth/me \
  -H "Authorization: Bearer <token>"
```

---

## Known Limitations

### Phase 5 Scope

1. **No User Menu Yet**: Logout button and user menu UI are planned for Phase 6
2. **No Session Expiration UI**: Token expires silently; user must refresh to see login page
3. **No "Forgot Password"**: Password reset is a future enhancement
4. **No Admin User Management UI**: User CRUD operations are planned for Phase 6

### Workarounds for Testing

**Manual Logout**:
```javascript
// In browser console
localStorage.removeItem('auth_token');
window.location.href = '/login';
```

**Manual User Creation** (for testing multiple users):
```bash
# Use backend API directly
curl -X POST http://localhost:9090/api/v1/users \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@test.com",
    "display_name": "Test User",
    "is_admin": false
  }'
```

---

## Troubleshooting

### Issue: Redirect Loop

**Symptoms**: Page keeps redirecting between `/` and `/login`

**Solutions**:
1. Check if token exists but is invalid:
   ```javascript
   localStorage.removeItem('auth_token');
   ```
2. Verify backend auth endpoints are accessible
3. Check browser console for errors

### Issue: Token Not Persisting

**Symptoms**: User logged out after page refresh

**Solutions**:
1. Verify localStorage is enabled in browser
2. Check browser privacy settings (localStorage may be disabled)
3. Ensure token is being saved:
   ```javascript
   localStorage.getItem('auth_token')
   ```

### Issue: CORS Errors

**Symptoms**: API requests fail with CORS errors

**Solutions**:
1. Ensure backend CORS is configured for `http://localhost:5173`
2. Check backend logs for CORS-related errors
3. Verify `api_app.py` has proper CORS middleware

### Issue: 401 Unauthorized After Login

**Symptoms**: API requests return 401 even after successful login

**Solutions**:
1. Verify token is in Authorization header:
   - Open DevTools → Network → Select request → Headers
2. Check token is valid (not expired)
3. Ensure backend secret key matches between login and subsequent requests

---

## Success Criteria

Phase 5 is considered successful when:

- ✅ Frontend builds without errors
- ✅ All TypeScript checks pass
- ✅ All ESLint checks pass
- ✅ All Prettier checks pass
- ✅ No circular dependencies detected
- ✅ Administrator setup flow works end-to-end
- ✅ Login flow works end-to-end
- ✅ Token persistence works across sessions
- ✅ Protected routes redirect to login when unauthenticated
- ✅ Authorization headers are added to API requests
- ✅ Password validation works correctly
- ✅ Error handling displays appropriate messages

---

## Next Steps (Phase 6)

Phase 6 will implement frontend UI updates including:
- User menu with logout button
- Admin indicators in UI
- Model management access control
- Queue filtering by user
- Session expiration handling
- Toast notifications for auth events

---

## Appendix A: Component API Reference

### AuthSlice

**State Shape**:
```typescript
interface AuthState {
  isAuthenticated: boolean;
  token: string | null;
  user: User | null;
  isLoading: boolean;
}
```

**Actions**:
- `setCredentials({ token, user })` - Store auth credentials
- `logout()` - Clear auth credentials
- `setLoading(boolean)` - Update loading state

**Selectors**:
- `selectIsAuthenticated(state)` - Get authentication status
- `selectCurrentUser(state)` - Get current user
- `selectAuthToken(state)` - Get token
- `selectIsAuthLoading(state)` - Get loading state

### Auth API Hooks

```typescript
// Login
const [login, { isLoading, error }] = useLoginMutation();
await login({ email, password, remember_me }).unwrap();

// Logout
const [logout] = useLogoutMutation();
await logout().unwrap();

// Get current user
const { data: user, isLoading, error } = useGetCurrentUserQuery();

// Setup
const [setup, { isLoading, error }] = useSetupMutation();
await setup({ email, display_name, password }).unwrap();
```

---

## Appendix B: File Locations

### Frontend Files Created
- `src/features/auth/store/authSlice.ts` - Redux slice
- `src/features/auth/components/LoginPage.tsx` - Login UI
- `src/features/auth/components/AdministratorSetup.tsx` - Setup UI
- `src/features/auth/components/ProtectedRoute.tsx` - Route wrapper
- `src/services/api/endpoints/auth.ts` - API endpoints

### Frontend Files Modified
- `src/app/components/InvokeAIUI.tsx` - Added BrowserRouter
- `src/app/components/App.tsx` - Added routing
- `src/app/store/store.ts` - Registered auth slice
- `src/services/api/index.ts` - Added auth headers
- `package.json` - Added react-router-dom
- `knip.ts` - Added auth files to ignore list

---

*Document Version: 1.0*  
*Last Updated: January 10, 2026*  
*Author: GitHub Copilot*
