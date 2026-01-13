# Phase 7 Testing Guide - Multiuser Authentication System

## Overview

This guide provides comprehensive testing instructions for Phase 7 of the multiuser implementation, which focuses on testing and security validation of the authentication system.

## Test Suite Organization

The Phase 7 test suite is organized into four main categories:

### 1. Unit Tests (`tests/app/services/auth/`)

#### Password Utilities Tests (`test_password_utils.py`)
- **Password Hashing Tests** (7 tests)
  - Hash generation with different salts
  - Special characters and Unicode support
  - Empty and very long passwords
  - Newline handling
  
- **Password Verification Tests** (9 tests)
  - Correct and incorrect password verification
  - Case sensitivity
  - Whitespace sensitivity
  - Special characters and Unicode
  - Invalid hash format handling
  
- **Password Strength Validation Tests** (12 tests)
  - Minimum length requirements
  - Uppercase, lowercase, and digit requirements
  - Special character handling
  - Unicode support
  - Edge cases (empty, very long)
  
- **Security Properties Tests** (3 tests)
  - Timing attack resistance
  - Hash randomization (salt uniqueness)
  - bcrypt format validation

**Total: 31 tests**

#### Token Service Tests (`test_token_service.py`)
- **Token Creation Tests** (5 tests)
  - Basic token creation
  - Custom expiration handling
  - Admin user tokens
  - Data preservation
  - Token uniqueness
  
- **Token Verification Tests** (6 tests)
  - Valid token verification
  - Invalid and malformed tokens
  - Expired token handling
  - Modified payload detection
  - Admin status preservation
  
- **Token Expiration Tests** (3 tests)
  - Fresh token validity
  - Long expiration periods
  - Short but valid expiration
  
- **Token Data Model Tests** (3 tests)
  - TokenData creation
  - Admin user handling
  - Model serialization
  
- **Token Security Tests** (3 tests)
  - Signature verification
  - Admin privilege forgery prevention
  - Algorithm security (HS256)

**Total: 20 tests**

### 2. Security Tests (`test_security.py`)

#### SQL Injection Prevention Tests (3 tests)
- Email field injection attempts
- Password field injection attempts
- User service injection protection

#### Authorization Bypass Tests (4 tests)
- Protected endpoint access without token
- Invalid token rejection
- Token forgery prevention
- Regular user privilege escalation prevention

#### Session Security Tests (2 tests)
- Token expiration validation
- Logout session invalidation

#### Input Validation Tests (3 tests)
- Email format validation
- XSS prevention in user data
- Path traversal prevention

#### Rate Limiting Tests (1 test, skipped)
- Login attempt rate limiting (documented for future implementation)

**Total: 13 tests**

### 3. Integration Tests (`test_data_isolation.py`)

#### Board Data Isolation Tests (3 tests)
- User can only see own boards
- Cannot access other user's boards
- Admin can see all boards

#### Image Data Isolation Tests (1 test)
- User image isolation (documented)

#### Workflow Data Isolation Tests (1 test)
- User workflow isolation (documented)

#### Queue Data Isolation Tests (1 test)
- User queue item isolation (documented)

#### Shared Board Tests (1 test, skipped)
- Shared board access (for future implementation)

#### Admin Authorization Tests (2 tests)
- Regular user cannot create admin
- Regular user cannot list all users

#### Data Integrity Tests (2 tests)
- User deletion cascades to owned data
- Concurrent operations maintain isolation

**Total: 11 tests**

### 4. Performance Tests (`test_performance.py`)

#### Password Performance Tests (3 tests)
- Hashing performance (10-500ms per hash)
- Verification performance (10-500ms per verification)
- Concurrent password operations

#### Token Performance Tests (3 tests)
- Creation performance (< 1ms per token)
- Verification performance (< 1ms per verification)
- Concurrent token operations (> 1000 ops/sec)

#### Authentication Overhead Tests (2 tests)
- Complete login flow performance (< 500ms)
- Token verification overhead (< 0.1ms per request)

#### User Service Performance Tests (3 tests)
- User creation performance (< 500ms)
- User lookup performance (< 5ms)
- User listing performance (< 50ms for 50 users)

#### Concurrent Sessions Tests (1 test)
- Multiple concurrent logins (< 10s for 20 users)

#### Scalability Benchmarks (1 test, marked slow)
- Authentication under sustained load (> 95% success rate)

**Total: 13 tests (1 marked slow)**

## Running the Tests

### Prerequisites

Ensure you have the development environment set up:

```bash
# Install dependencies
pip install -e ".[dev,test]"
```

### Running All Phase 7 Tests

```bash
# Run all auth service tests
pytest tests/app/services/auth/ -v

# Run with coverage
pytest tests/app/services/auth/ --cov=invokeai.app.services.auth --cov-report=html
```

### Running Specific Test Categories

```bash
# Unit tests only
pytest tests/app/services/auth/test_password_utils.py -v
pytest tests/app/services/auth/test_token_service.py -v

# Security tests
pytest tests/app/services/auth/test_security.py -v

# Integration tests
pytest tests/app/services/auth/test_data_isolation.py -v

# Performance tests (fast only)
pytest tests/app/services/auth/test_performance.py -v

# Performance tests (including slow benchmarks)
pytest tests/app/services/auth/test_performance.py -v -m slow
```

### Running Individual Tests

```bash
# Run a specific test class
pytest tests/app/services/auth/test_password_utils.py::TestPasswordHashing -v

# Run a specific test method
pytest tests/app/services/auth/test_password_utils.py::TestPasswordHashing::test_hash_password_returns_different_hash_each_time -v
```

## Expected Test Results

### Test Coverage Goals

- **Password utilities**: 100% coverage
- **Token service**: 100% coverage
- **Security tests**: Comprehensive attack vector coverage
- **Integration tests**: Core isolation scenarios covered
- **Performance tests**: Baseline metrics established

### Performance Benchmarks

Expected performance metrics on modern hardware:

| Operation | Expected Performance |
|-----------|---------------------|
| Password hashing | 50-100ms per hash |
| Password verification | 50-100ms per verification |
| Token creation | < 1ms per token |
| Token verification | < 1ms per verification |
| Complete login flow | < 500ms |
| User lookup | < 5ms |
| Token verification overhead | < 0.1ms per request |
| Concurrent token ops | > 1000 ops/second |

### Security Test Expectations

All security tests should pass with no vulnerabilities detected:

✅ SQL injection prevented (parameterized queries)
✅ Authorization bypass prevented (token signature verification)
✅ XSS prevented (proper data escaping)
✅ Path traversal prevented (input validation)
✅ Token forgery prevented (HMAC signature)
✅ Admin privilege escalation prevented (token validation)

## Manual Testing Procedures

### 1. SQL Injection Testing

**Manual Test Cases:**

```bash
# Test login with SQL injection in email field
curl -X POST http://localhost:9090/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"'\'' OR '\''1'\''='\''1","password":"test","remember_me":false}'

# Expected: 401 Unauthorized (not 500 or 200)
```

### 2. Token Security Testing

**Manual Test Cases:**

```bash
# 1. Try accessing protected endpoint without token
curl http://localhost:9090/api/v1/auth/me
# Expected: 401 Unauthorized

# 2. Try accessing with invalid token
curl -H "Authorization: Bearer invalid_token" \
  http://localhost:9090/api/v1/auth/me
# Expected: 401 Unauthorized

# 3. Try modifying token payload
# (Modify a character in the token string)
curl -H "Authorization: Bearer eyJhbGc...modified..." \
  http://localhost:9090/api/v1/auth/me
# Expected: 401 Unauthorized
```

### 3. Password Security Testing

**Manual Test Cases:**

1. **Weak Password Rejection:**
   ```bash
   # Try creating user with weak password
   curl -X POST http://localhost:9090/api/v1/auth/setup \
     -H "Content-Type: application/json" \
     -d '{"email":"admin@test.com","display_name":"Admin","password":"weak"}'
   # Expected: 400 Bad Request with password requirement message
   ```

2. **Special Characters:**
   ```bash
   # Try password with special characters
   curl -X POST http://localhost:9090/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@test.com","password":"Test!@#$%123","remember_me":false}'
   # Expected: 200 OK if user exists with this password
   ```

### 4. Data Isolation Testing

**Manual Test Cases:**

1. **Board Isolation:**
   - Login as User A, create a board, note the board_id
   - Login as User B, try to list boards
   - Verify User B cannot see User A's board

2. **Token Isolation:**
   - Login as User A, save token
   - Login as User B, save token
   - Try using User A's token to access data - should only see User A's data
   - Try using User B's token to access data - should only see User B's data

### 5. Performance Testing

**Manual Benchmarking:**

```bash
# Use Apache Bench or similar tool to test login performance
ab -n 100 -c 10 -p login.json -T "application/json" \
  http://localhost:9090/api/v1/auth/login

# Where login.json contains:
# {"email":"test@example.com","password":"TestPass123","remember_me":false}
```

## Troubleshooting

### Common Issues

#### 1. Tests Fail Due to Missing Dependencies

```bash
# Solution: Install test dependencies
pip install -e ".[dev,test]"
```

#### 2. Database Lock Errors

```bash
# Solution: Use in-memory database for tests
# Tests are already configured to use in-memory SQLite
```

#### 3. Slow Password Tests

```bash
# This is expected - bcrypt is intentionally slow (50-100ms)
# If tests are timing out, increase timeout values
pytest tests/app/services/auth/ --timeout=30
```

#### 4. Performance Tests Fail on Slow Hardware

```bash
# Performance expectations may need adjustment for different hardware
# Check actual times in test output and adjust assertions if needed
```

## Security Checklist

Before completing Phase 7, verify:

- [ ] All SQL injection tests pass
- [ ] All authorization bypass tests pass
- [ ] Token forgery prevention works
- [ ] Password hashing uses bcrypt with proper salt
- [ ] Tokens use HMAC signature with secure algorithm (HS256)
- [ ] Password strength validation enforces requirements
- [ ] Data isolation tests confirm user separation
- [ ] No sensitive data in logs or error messages
- [ ] Token expiration is properly enforced
- [ ] All security tests pass with no failures

## Coverage Report

To generate a coverage report:

```bash
# Generate HTML coverage report
pytest tests/app/services/auth/ --cov=invokeai.app.services.auth --cov-report=html

# Open the report
open htmlcov/index.html
```

Target coverage: **> 90%** for all auth modules

## Integration with Existing Tests

Phase 7 tests complement existing tests:

- **Phase 3 tests** (`test_auth.py`): API endpoint integration tests
- **Phase 4 tests** (`test_user_service.py`): User service unit tests
- **Phase 6 tests** (`test_boards_multiuser.py`): Board multiuser tests

All tests should pass together:

```bash
# Run all auth-related tests
pytest tests/app/routers/test_auth.py \
       tests/app/services/users/ \
       tests/app/services/auth/ \
       tests/app/routers/test_boards_multiuser.py \
       -v
```

## Next Steps

After Phase 7 testing is complete:

1. **Review test results** and address any failures
2. **Generate coverage report** and improve coverage if needed
3. **Run security audit** using findings from security tests
4. **Document any discovered issues** in the issue tracker
5. **Prepare for Phase 8** (Documentation) or Phase 9 (Migration Support)

## Test Summary

| Category | Test Count | Status |
|----------|-----------|--------|
| Password Utils | 31 | ✅ Comprehensive |
| Token Service | 20 | ✅ Comprehensive |
| Security | 13 | ✅ Comprehensive |
| Data Isolation | 11 | ✅ Core scenarios |
| Performance | 13 | ✅ Benchmarks set |
| **Total** | **88** | ✅ **Phase 7 Complete** |

## References

- Implementation Plan: `docs/multiuser/implementation_plan.md`
- Specification: `docs/multiuser/specification.md`
- Phase 3 Testing: `docs/multiuser/phase3_testing.md`
- Phase 4 Testing: `docs/multiuser/phase4_verification.md`
- Phase 5 Testing: `docs/multiuser/phase5_testing.md`
- Phase 6 Testing: `docs/multiuser/phase6_testing.md`
