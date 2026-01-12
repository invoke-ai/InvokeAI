# Testing Token Expiration

This guide explains how to test JWT token expiration without waiting for the full expiration period (7 days for "Remember me" tokens).

## Methods for Testing Token Expiration

### Method 1: Modify Backend Token Expiration (Recommended)

The backend JWT token expiration is configured in the authentication service. You can temporarily modify the expiration time for testing purposes.

**Location**: `invokeai/app/services/auth/auth_service.py` (or similar auth configuration file)

**Steps**:
1. Find the JWT token expiration configuration in the backend code
2. Change the expiration time from 7 days to a shorter period (e.g., 2 minutes):
   ```python
   # For remember_me=True tokens
   expires_delta = timedelta(minutes=2)  # Instead of days=7
   
   # For regular tokens
   expires_delta = timedelta(minutes=1)  # Instead of minutes=30
   ```
3. Restart the backend server
4. Log in with "Remember me" checked
5. Wait 2 minutes and verify that:
   - The token expires and you're redirected to login
   - API requests return 401 Unauthorized
   - The app handles expiration gracefully

**Remember to revert these changes after testing!**

### Method 2: Manually Expire Token in Browser

You can manually test token expiration by modifying or deleting the token from localStorage:

**Steps**:
1. Log in to the application
2. Open browser DevTools (F12)
3. Go to Application/Storage → Local Storage → `http://localhost:5173`
4. Find the `auth_token` key
5. **Option A**: Delete the token completely
   - Click on `auth_token` and press Delete
   - Refresh the page
   - You should be redirected to login
6. **Option B**: Replace with an expired/invalid token
   - Edit the `auth_token` value to invalid characters (e.g., "invalid-token")
   - Refresh the page
   - The app should detect invalid token and redirect to login

### Method 3: Use Backend Admin Tools

If the backend provides admin tools or API endpoints to invalidate tokens:

1. Log in and note your token (from localStorage)
2. Use admin API to invalidate/blacklist the token
3. Try to make an authenticated request
4. Verify the app handles the invalid token gracefully

### Method 4: Modify Token Payload (Advanced)

For testing JWT token structure issues:

1. Copy the token from localStorage
2. Decode it using a JWT debugger (jwt.io)
3. Modify the `exp` (expiration) claim to a past timestamp
4. Re-encode the token (note: this requires the secret key, so this only works if you control the backend)
5. Replace the token in localStorage
6. Test the application behavior

## Expected Behavior on Token Expiration

When a token expires, the application should:

1. **On API Request**: Return 401 Unauthorized error
2. **Frontend Handling**: 
   - The `ProtectedRoute` component detects the error
   - Calls `logout()` to clear auth state
   - Removes token from localStorage
   - Redirects user to `/login`
3. **Websocket**: Connection should fail with auth error
4. **User Experience**: Clean redirect to login page with no data loss (draft workflow, settings, etc. should persist)

## Testing Checklist

- [ ] Token expires after configured time period
- [ ] Expired token is detected on next page load
- [ ] Expired token is detected during API requests
- [ ] User is redirected to login page gracefully
- [ ] No infinite redirect loops occur
- [ ] Auth state is properly cleared
- [ ] Token is removed from localStorage
- [ ] User can log in again successfully
- [ ] Websocket connection fails appropriately with expired token
- [ ] Error messages are user-friendly

## Configuration Reference

The token expiration is controlled by these JWT settings in the backend:

```python
# Standard login token (30 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# "Remember me" token (7 days)
REMEMBER_ME_TOKEN_EXPIRE_DAYS = 7
```

For testing, you can create environment variables or configuration options:
```bash
# .env file for testing
AUTH_TOKEN_EXPIRE_MINUTES=2  # Short expiration for testing
```

## Debugging Tips

### Check Token in DevTools
```javascript
// In browser console
const token = localStorage.getItem('auth_token');
console.log('Token:', token);

// Decode token (without verification)
const parts = token.split('.');
const payload = JSON.parse(atob(parts[1]));
console.log('Payload:', payload);
console.log('Expires:', new Date(payload.exp * 1000));
console.log('Is Expired:', Date.now() > payload.exp * 1000);
```

### Watch for Token Expiration
You can add a temporary debug script to monitor token status:
```javascript
// In browser console
setInterval(() => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    const parts = token.split('.');
    const payload = JSON.parse(atob(parts[1]));
    const expiresIn = Math.floor((payload.exp * 1000 - Date.now()) / 1000);
    console.log(`Token expires in ${expiresIn} seconds`);
  }
}, 10000); // Check every 10 seconds
```

### Backend Logs
Monitor backend logs for authentication failures:
```bash
# Look for JWT decode errors, expired token errors, etc.
tail -f invokeai.log | grep -i "auth\|token\|jwt"
```

## Conclusion

For routine testing, **Method 1** (modifying backend expiration time) is the most realistic and thorough approach. For quick smoke tests, **Method 2** (manually deleting/modifying localStorage) is fastest.

Always test the complete flow:
1. Login → Token stored
2. Use app → API calls succeed
3. Token expires → API calls fail with 401
4. Frontend detects → Redirect to login
5. Login again → New token, full functionality restored
