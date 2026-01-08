# Phase 3: Authentication Middleware - Functional Testing Guide

## Overview

Phase 3 of the multiuser implementation adds authentication middleware and endpoints to InvokeAI. This document provides comprehensive testing instructions to validate the implementation.

## Prerequisites

1. **Development Environment Setup**
   ```bash
   # Install development dependencies
   pip install -e ".[dev,test]"
   ```

2. **Start InvokeAI in Development Mode**
   ```bash
   python -m invokeai.app.run_app --dev_reload
   ```
   The server should start on `http://localhost:9090`

## Automated Testing

### Running Unit Tests

The Phase 3 implementation includes comprehensive integration tests for all authentication endpoints.

```bash
# Run all auth router tests
pytest tests/app/routers/test_auth.py -v

# Run specific test
pytest tests/app/routers/test_auth.py::test_login_success -v

# Run with coverage
pytest tests/app/routers/test_auth.py --cov=invokeai.app.api.routers.auth --cov-report=html
```

### Test Coverage

The test suite covers:
- ✅ User login with valid credentials
- ✅ User login with "remember me" flag (7-day token expiration)
- ✅ Login failure with invalid password
- ✅ Login failure with nonexistent user
- ✅ Login failure with inactive user account
- ✅ User logout (stateless JWT)
- ✅ Getting current user information
- ✅ Initial admin setup
- ✅ Admin setup validation (prevents duplicate admins)
- ✅ Password strength validation
- ✅ Token validation and authentication
- ✅ Admin flag in JWT tokens

## Manual Testing

### 1. Testing Initial Admin Setup

**Test Case:** Create the first admin user

1. **Ensure no admin exists** (fresh database recommended)

2. **Call the setup endpoint:**
   ```bash
   curl -X POST http://localhost:9090/api/v1/auth/setup \
     -H "Content-Type: application/json" \
     -d '{
       "email": "admin@invokeai.local",
       "display_name": "Admin User",
       "password": "AdminPass123"
     }'
   ```

3. **Expected Response (200 OK):**
   ```json
   {
     "success": true,
     "user": {
       "user_id": "some-uuid",
       "email": "admin@invokeai.local",
       "display_name": "Admin User",
       "is_admin": true,
       "is_active": true,
       "created_at": "2026-01-08T...",
       "updated_at": "2026-01-08T...",
       "last_login_at": null
     }
   }
   ```

4. **Verify admin cannot be created again:**
   ```bash
   curl -X POST http://localhost:9090/api/v1/auth/setup \
     -H "Content-Type: application/json" \
     -d '{
       "email": "admin2@invokeai.local",
       "display_name": "Second Admin",
       "password": "AdminPass123"
     }'
   ```

5. **Expected Response (400 Bad Request):**
   ```json
   {
     "detail": "Administrator account already configured"
   }
   ```

### 2. Testing User Login

**Test Case:** Authenticate with valid credentials

1. **Login with valid credentials:**
   ```bash
   curl -X POST http://localhost:9090/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{
       "email": "admin@invokeai.local",
       "password": "AdminPass123",
       "remember_me": false
     }'
   ```

2. **Expected Response (200 OK):**
   ```json
   {
     "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
     "user": {
       "user_id": "some-uuid",
       "email": "admin@invokeai.local",
       "display_name": "Admin User",
       "is_admin": true,
       "is_active": true,
       ...
     },
     "expires_in": 86400
   }
   ```

3. **Save the token** for subsequent requests (replace `YOUR_TOKEN` below)

### 3. Testing Token Validation

**Test Case:** Access protected endpoints with token

1. **Get current user information:**
   ```bash
   curl -X GET http://localhost:9090/api/v1/auth/me \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

2. **Expected Response (200 OK):**
   ```json
   {
     "user_id": "some-uuid",
     "email": "admin@invokeai.local",
     "display_name": "Admin User",
     "is_admin": true,
     "is_active": true,
     ...
   }
   ```

3. **Test without token (should fail):**
   ```bash
   curl -X GET http://localhost:9090/api/v1/auth/me
   ```

4. **Expected Response (401 Unauthorized):**
   ```json
   {
     "detail": "Missing authentication credentials"
   }
   ```

### 4. Testing Invalid Credentials

**Test Case:** Login with wrong password

1. **Attempt login with wrong password:**
   ```bash
   curl -X POST http://localhost:9090/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{
       "email": "admin@invokeai.local",
       "password": "WrongPassword",
       "remember_me": false
     }'
   ```

2. **Expected Response (401 Unauthorized):**
   ```json
   {
     "detail": "Incorrect email or password"
   }
   ```

### 5. Testing "Remember Me" Feature

**Test Case:** Verify extended token expiration

1. **Login with remember_me=true:**
   ```bash
   curl -X POST http://localhost:9090/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{
       "email": "admin@invokeai.local",
       "password": "AdminPass123",
       "remember_me": true
     }'
   ```

2. **Verify expires_in is 604800 (7 days):**
   ```json
   {
     "token": "...",
     "user": {...},
     "expires_in": 604800
   }
   ```

### 6. Testing Logout

**Test Case:** User logout (stateless, client-side operation)

1. **Call logout endpoint:**
   ```bash
   curl -X POST http://localhost:9090/api/v1/auth/logout \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

2. **Expected Response (200 OK):**
   ```json
   {
     "success": true
   }
   ```

   **Note:** Since we use stateless JWT tokens, logout is currently a no-op on the server side. The client should discard the token. Future implementations may add token blacklisting.

### 7. Testing Password Validation

**Test Case:** Weak password should be rejected

1. **Attempt setup with weak password:**
   ```bash
   curl -X POST http://localhost:9090/api/v1/auth/setup \
     -H "Content-Type: application/json" \
     -d '{
       "email": "admin@invokeai.local",
       "display_name": "Admin User",
       "password": "weak"
     }'
   ```

2. **Expected Response (400 Bad Request):**
   ```json
   {
     "detail": "Password must be at least 8 characters long"
   }
   ```

## Testing with OpenAPI/Swagger UI

InvokeAI includes interactive API documentation that can be used for testing:

1. **Open Swagger UI:**
   Navigate to `http://localhost:9090/docs`

2. **Test the setup endpoint:**
   - Find `POST /api/v1/auth/setup` in the API list
   - Click "Try it out"
   - Enter the request body and execute
   - Review the response

3. **Test authentication flow:**
   - Call `POST /api/v1/auth/login`
   - Copy the returned token
   - Click "Authorize" button (🔓 icon at top)
   - Enter: `Bearer YOUR_TOKEN`
   - Now you can test protected endpoints like `GET /api/v1/auth/me`

## Security Testing

### 1. Token Expiration

**Test Case:** Verify tokens expire correctly

1. Generate a token with short expiration (modify `TOKEN_EXPIRATION_NORMAL` in code for testing)
2. Wait for expiration time to pass
3. Attempt to use expired token
4. Expected: 401 Unauthorized with "Invalid or expired authentication token"

### 2. Invalid Token Format

**Test Case:** Malformed tokens should be rejected

```bash
curl -X GET http://localhost:9090/api/v1/auth/me \
  -H "Authorization: Bearer invalid_token_format"
```

Expected: 401 Unauthorized

### 3. SQL Injection Prevention

**Test Case:** Malicious input should be sanitized

```bash
curl -X POST http://localhost:9090/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@invokeai.local OR 1=1--",
    "password": "anything",
    "remember_me": false
  }'
```

Expected: 401 Unauthorized (not SQL error)

## Database Verification

### Verify Users Table Created

```bash
# Connect to SQLite database
sqlite3 invokeai.db

# Check users table structure
.schema users

# List all users
SELECT user_id, email, display_name, is_admin, is_active FROM users;

# Exit
.quit
```

### Expected Schema

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
```

## Common Issues and Troubleshooting

### Issue: "No module named 'passlib'"

**Solution:** Install authentication dependencies
```bash
pip install passlib[bcrypt] python-jose[cryptography]
```

### Issue: "users service not found"

**Solution:** Ensure the users service is registered in the invoker. Check `api/dependencies.py` initialization.

### Issue: Migration fails

**Solution:** Check migration 25 is registered in `sqlite_util.py` and run:
```bash
python -m invokeai.app.migrate
```

### Issue: Token always returns 401

**Solution:** 
1. Verify SECRET_KEY is consistent between token creation and validation
2. Check system time is correct (JWT uses timestamp validation)
3. Verify token isn't expired

## Test Results Checklist

Use this checklist to verify Phase 3 implementation:

- [ ] Migration 25 creates users table successfully
- [ ] Initial admin setup works (POST /api/v1/auth/setup)
- [ ] Cannot create second admin via setup endpoint
- [ ] User login works with valid credentials
- [ ] User login fails with invalid credentials
- [ ] User login fails with nonexistent user
- [ ] Token includes correct user information
- [ ] Remember me provides 7-day expiration
- [ ] Normal login provides 1-day expiration
- [ ] Protected endpoints require Bearer token
- [ ] GET /api/v1/auth/me returns current user
- [ ] Logout endpoint responds successfully
- [ ] Invalid tokens are rejected (401)
- [ ] Missing tokens are rejected (401)
- [ ] Password validation enforces strength requirements
- [ ] Admin flag is correctly stored and returned
- [ ] All automated tests pass

## Performance Testing

### Token Generation Performance

```bash
# Time multiple token generations
time for i in {1..100}; do
  curl -s -X POST http://localhost:9090/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"admin@invokeai.local","password":"AdminPass123","remember_me":false}' \
    > /dev/null
done
```

Expected: < 5 seconds for 100 logins (avg ~50ms per login)

### Token Validation Performance

```bash
# Get a token first
TOKEN=$(curl -s -X POST http://localhost:9090/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@invokeai.local","password":"AdminPass123","remember_me":false}' | jq -r .token)

# Time multiple validations
time for i in {1..100}; do
  curl -s -X GET http://localhost:9090/api/v1/auth/me \
    -H "Authorization: Bearer $TOKEN" \
    > /dev/null
done
```

Expected: < 3 seconds for 100 validations (avg ~30ms per validation)

## Success Criteria

Phase 3 is complete when:

✅ All automated tests pass  
✅ All manual test cases succeed  
✅ Security tests show no vulnerabilities  
✅ Performance meets targets  
✅ Database schema is correct  
✅ API documentation is accurate  
✅ No regressions in existing functionality  

## Next Steps

After Phase 3 is validated:

1. **Phase 4:** Update existing services for multi-tenancy (boards, images, workflows)
2. **Phase 5:** Frontend authentication integration
3. **Phase 6:** UI updates for multi-user features

## Support

For issues or questions about Phase 3 implementation:
- Check the [Implementation Plan](implementation_plan.md)
- Review the [Specification](specification.md)
- Create a GitHub issue with the `multiuser` label
