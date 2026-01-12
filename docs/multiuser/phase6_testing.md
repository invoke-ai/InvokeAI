# Phase 6 Testing Guide - Frontend UI Updates

## Overview

This document provides comprehensive testing instructions for Phase 6 of the multiuser implementation, which includes:
1. UserMenu component with logout functionality
2. Admin badge display
3. Model Manager access restrictions for non-admin users

## Prerequisites

### Backend Setup
1. Ensure the backend server is running with Phase 1-4 multiuser features:
   ```bash
   cd /path/to/InvokeAI
   python -m invokeai.app.run_app
   ```

2. Backend should be accessible at: `http://localhost:9090`

### Test User Setup

You'll need at least one admin user and one regular (non-admin) user for testing.

**Option 1: Use the provided script (Recommended)**

Add a regular user:
```bash
cd /path/to/InvokeAI
python scripts/add_user.py --email testuser@test.local --password TestPass123 --name "Test User"
```

Add an admin user:
```bash
python scripts/add_user.py --email admin@test.local --password AdminPass123 --name "Admin User" --admin
```

Interactive mode (prompts for details):
```bash
python scripts/add_user.py
```

**Option 2: Direct SQL (Advanced)**

If you need to add a user directly to the database:
```bash
# Find your database file
python -c "from invokeai.app.services.config import InvokeAIAppConfig; print(InvokeAIAppConfig.get_config().db_path)"

# Use sqlite3 to add a user (requires password hash)
sqlite3 /path/to/invokeai.db
```

Note: The script in Option 1 handles password hashing and validation automatically.

### Frontend Setup
1. Install dependencies:
   ```bash
   cd invokeai/frontend/web
   pnpm install
   ```

2. Start the development server:
   ```bash
   pnpm dev
   ```

3. Frontend should be accessible at: `http://localhost:5173`

## Test Scenarios

### Test 1: UserMenu Display for Admin User

**Objective:** Verify UserMenu displays correctly for administrator users

**Steps:**
1. Log in as an administrator user
2. Look at the vertical navigation bar on the left side
3. Locate the user icon button (above the bell icon)

**Expected Results:**
- ✅ User icon button is visible in the vertical navigation bar
- ✅ User icon is positioned above the Notifications icon
- ✅ User icon has a tooltip showing "User Menu"

**Test Data:**
- Admin email: (use your administrator account)
- Admin password: (use your administrator password)

---

### Test 2: UserMenu Contents for Admin User

**Objective:** Verify UserMenu dropdown shows correct information for admin users

**Steps:**
1. Log in as an administrator user
2. Click the user icon in the vertical navigation bar
3. Inspect the dropdown menu contents

**Expected Results:**
- ✅ Dropdown menu appears below the user icon
- ✅ User's display name is shown (or email if no display name)
- ✅ User's email is shown below the display name
- ✅ "Admin" badge is visible with yellow/gold color scheme
- ✅ "Logout" menu item is present with a sign-out icon

**Screenshot Location:** `docs/multiuser/screenshots/phase6_usermenu_admin.png`

---

### Test 3: UserMenu Contents for Regular User

**Objective:** Verify UserMenu dropdown shows correct information for non-admin users

**Steps:**
1. Create a regular (non-admin) user account via admin panel
2. Log out of admin account
3. Log in as the regular user
4. Click the user icon in the vertical navigation bar
5. Inspect the dropdown menu contents

**Expected Results:**
- ✅ Dropdown menu appears below the user icon
- ✅ User's display name is shown (or email if no display name)
- ✅ User's email is shown below the display name
- ✅ "Admin" badge is NOT visible
- ✅ "Logout" menu item is present with a sign-out icon

**Screenshot Location:** `docs/multiuser/screenshots/phase6_usermenu_regular.png`

---

### Test 4: Logout Functionality

**Objective:** Verify logout button correctly logs out the user

**Steps:**
1. Log in as any user (admin or regular)
2. Click the user icon in the vertical navigation bar
3. Click the "Logout" menu item
4. Observe the behavior

**Expected Results:**
- ✅ Backend logout API is called (`POST /api/v1/auth/logout`)
- ✅ User is redirected to the login page (`/login`)
- ✅ Auth token is removed from localStorage
- ✅ User cannot access protected routes without logging in again
- ✅ Attempting to navigate to `/` redirects to `/login`

**Verification Commands (Browser Console):**
```javascript
// Before logout
console.log(localStorage.getItem('auth_token')); // Should show token

// After logout
console.log(localStorage.getItem('auth_token')); // Should be null
```

---

### Test 5: Model Manager Tab - Admin Access

**Objective:** Verify admin users can access the Model Manager tab

**Steps:**
1. Log in as an administrator user
2. Look at the vertical navigation bar
3. Locate the cube icon (Model Manager tab button)
4. Click the cube icon
5. Observe the content area

**Expected Results:**
- ✅ Model Manager tab button (cube icon) is visible in the navigation bar
- ✅ Model Manager tab button is positioned above the Queue tab
- ✅ Clicking the button switches to the Model Manager tab
- ✅ Model Manager interface is displayed with:
  - Model list panel on the left
  - Model details panel on the right
  - "Add Models" button when a model is selected

**Screenshot Location:** `docs/multiuser/screenshots/phase6_models_admin.png`

---

### Test 6: Model Manager Tab - Non-Admin Restriction (Hidden Button)

**Objective:** Verify non-admin users do not see the Model Manager tab button

**Steps:**
1. Log in as a regular (non-admin) user
2. Look at the vertical navigation bar
3. Search for the cube icon (Model Manager tab button)

**Expected Results:**
- ✅ Model Manager tab button (cube icon) is NOT visible
- ✅ Navigation shows: Generate, Canvas, Upscaling, Workflows tabs
- ✅ Navigation shows Queue tab but NOT Models tab
- ✅ User can access all other tabs normally

**Screenshot Location:** `docs/multiuser/screenshots/phase6_navbar_regular.png`

---

### Test 7: Model Manager Tab - Non-Admin Direct URL Access

**Objective:** Verify non-admin users see access denied message if they navigate directly to Model Manager

**Steps:**
1. Log in as a regular (non-admin) user
2. In the browser address bar, manually navigate to: `http://localhost:5173/models`
   (or click on the Models tab if it somehow appears)

**Expected Results:**
- ✅ Page displays "Model Manager" heading
- ✅ Page displays access denied message: "This feature is only available to administrators."
- ✅ No model list or management interface is shown
- ✅ User cannot perform any model management actions

**Screenshot Location:** `docs/multiuser/screenshots/phase6_models_denied.png`

---

### Test 8: Logout Persistence After Browser Refresh

**Objective:** Verify logout state persists across browser refresh

**Steps:**
1. Log in as any user
2. Click logout
3. Verify you're on the login page
4. Press browser refresh (F5 or Cmd+R)

**Expected Results:**
- ✅ User remains on the login page
- ✅ No automatic login occurs
- ✅ User must re-enter credentials to access the app

---

### Test 9: UserMenu Styling and Responsiveness

**Objective:** Verify UserMenu UI elements are properly styled

**Steps:**
1. Log in as any user
2. Click the user icon
3. Inspect the visual appearance of the menu

**Expected Results:**
- ✅ User icon button has hover effect
- ✅ Dropdown menu has proper padding and spacing
- ✅ Text is legible and properly aligned
- ✅ Admin badge uses invokeYellow color scheme
- ✅ Logout menu item has hover effect
- ✅ Menu closes when clicking outside

**Visual Checks:**
- Font sizes are appropriate (display name: sm, email: xs)
- Colors match the app theme
- Admin badge is noticeable but not overwhelming
- Icons are properly sized and aligned

---

### Test 10: Accessibility Testing

**Objective:** Verify UserMenu is accessible via keyboard

**Steps:**
1. Log in as any user
2. Press Tab repeatedly to navigate through the interface
3. When user icon is focused, press Enter
4. Use arrow keys to navigate menu items
5. Press Enter on Logout

**Expected Results:**
- ✅ User icon can be focused with Tab key
- ✅ User icon has visible focus indicator
- ✅ Enter key opens the menu
- ✅ Arrow keys navigate menu items
- ✅ Enter key on Logout logs out the user
- ✅ Escape key closes the menu
- ✅ All interactive elements have aria-label attributes

---

## Browser Compatibility Testing

Test the following scenarios in multiple browsers:

### Supported Browsers
- Chrome 120+
- Firefox 120+
- Safari 17+
- Edge 120+

### Key Features to Verify
1. UserMenu dropdown appearance and positioning
2. Logout functionality
3. Model Manager access restrictions
4. Smooth navigation transitions

---

## Automated Testing

### Running Frontend Linters

```bash
cd invokeai/frontend/web

# Run all linters
pnpm lint

# Run individual linters
pnpm lint:eslint    # ESLint checks
pnpm lint:prettier  # Code formatting
pnpm lint:tsc       # TypeScript type checks
pnpm lint:knip      # Unused code detection
pnpm lint:dpdm      # Circular dependency detection
```

**Expected Results:**
- ✅ All linters pass with no errors
- ✅ No warnings (eslint uses --max-warnings=0)
- ✅ No circular dependencies

### Building the Frontend

```bash
cd invokeai/frontend/web
pnpm build
```

**Expected Results:**
- ✅ Build completes successfully
- ✅ No TypeScript errors
- ✅ Assets are generated in `dist/` directory
- ✅ Bundle size is reasonable (~700kB gzipped for main bundle)

---

## Integration Testing

### Test 11: Multi-Tab Session Management

**Objective:** Verify logout in one tab affects other tabs

**Steps:**
1. Log in as any user
2. Open the app in a second browser tab
3. In Tab 1, click logout
4. Switch to Tab 2 and try to perform an action

**Expected Results:**
- ✅ Tab 1 redirects to login page
- ✅ Tab 2's API calls fail with 401 Unauthorized
- ✅ Tab 2 should redirect to login page on next navigation

---

### Test 12: Rapid Logout Clicks

**Objective:** Verify logout handles rapid clicks gracefully

**Steps:**
1. Log in as any user
2. Open UserMenu
3. Click Logout button multiple times rapidly

**Expected Results:**
- ✅ No JavaScript errors in console
- ✅ Single logout API call is made (or duplicates are handled)
- ✅ User is redirected to login page only once
- ✅ No visual glitches or stuck states

---

## Performance Testing

### Test 13: UserMenu Performance

**Objective:** Verify UserMenu doesn't impact performance

**Checks:**
- ✅ UserMenu icon renders without delay
- ✅ Dropdown opens instantly (<100ms)
- ✅ Logout action is responsive
- ✅ No memory leaks (check with browser DevTools)

**Browser DevTools:**
1. Open Performance tab
2. Record a session including:
   - Opening UserMenu
   - Closing UserMenu
   - Logout
3. Check for:
   - No long tasks (>50ms)
   - No layout thrashing
   - Proper cleanup after logout

---

## Error Scenarios

### Test 14: Backend Logout Failure

**Objective:** Verify app handles backend logout errors gracefully

**Steps:**
1. Log in as any user
2. Stop the backend server
3. Click logout in the UserMenu

**Expected Results:**
- ✅ Frontend still removes token from localStorage
- ✅ User is redirected to login page
- ✅ No error dialogs or crashes
- ✅ Clean logout state

**Rationale:** Client-side logout should succeed even if backend is unavailable.

---

### Test 15: Missing User Data

**Objective:** Verify UserMenu handles edge cases

**Scenario 1: User with no display name**
- ✅ Shows email as primary text
- ✅ Still shows email as secondary text (duplicate is acceptable)

**Scenario 2: User with very long email**
- ✅ Text truncates with ellipsis (noOfLines={1})
- ✅ Dropdown width accommodates reasonably long text
- ✅ No horizontal scrolling

---

## Regression Testing

Verify that Phase 6 changes don't break existing functionality:

### Test 16: Other Navigation Elements

**Objective:** Verify other navigation buttons still work

**Steps:**
1. Log in as any user
2. Test all navigation buttons in order:
   - Generate tab
   - Canvas tab
   - Upscaling tab
   - Workflows tab
   - Queue tab
   - (Models tab - if admin)
   - Status indicator
   - Notifications
   - Videos modal
   - Settings menu

**Expected Results:**
- ✅ All buttons respond correctly
- ✅ All tabs load properly
- ✅ No layout shifts or overlapping elements
- ✅ Proper tab highlighting (active state)

---

### Test 17: Settings Menu Functionality

**Objective:** Verify Settings menu still works after adding UserMenu

**Steps:**
1. Log in as any user
2. Click the Settings (gear) icon
3. Navigate through settings panels
4. Make a settings change
5. Close settings modal

**Expected Results:**
- ✅ Settings modal opens correctly
- ✅ All settings panels are accessible
- ✅ Settings changes are saved
- ✅ No conflicts with UserMenu

---

## Security Testing

### Test 18: Direct Model Manager Access

**Objective:** Verify backend enforces admin-only model operations

**Steps:**
1. Log in as a regular user
2. Open browser DevTools → Network tab
3. Try to access Model Manager tab (should see access denied)
4. Manually craft an API request to backend model endpoints:
   ```javascript
   // In browser console
   fetch('/api/v1/models', {
     headers: {
       'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
     }
   }).then(r => r.json()).then(console.log)
   ```

**Expected Results:**
- ✅ Backend returns 403 Forbidden for model management endpoints
- ✅ Frontend cannot bypass restrictions via direct API calls
- ✅ Regular users cannot list, add, or delete models

**Note:** Backend authorization is the primary security layer; frontend restrictions are UX enhancements.

---

### Test 19: Token Removal on Logout

**Objective:** Verify no auth tokens remain after logout

**Steps:**
1. Log in as any user
2. Open DevTools → Application → Local Storage
3. Note the `auth_token` value
4. Click logout
5. Check Local Storage again

**Expected Results:**
- ✅ `auth_token` is removed from localStorage
- ✅ No other auth-related data persists
- ✅ Redux state is cleared (if persisted)

---

## Translation Testing

### Test 20: Localization Keys

**Objective:** Verify all UI text uses translation keys

**Translation Keys Used:**
- `auth.userMenu` - User Menu tooltip
- `auth.admin` - Admin badge
- `auth.logout` - Logout button
- `auth.adminOnlyFeature` - Access denied message
- `modelManager.modelManager` - Model Manager heading
- `ui.tabs.models` - Models tab label

**Steps:**
1. Verify all keys exist in `public/locales/en.json`
2. (Optional) Test with a different locale if available

**Expected Results:**
- ✅ All translation keys are defined
- ✅ Text displays correctly in the UI
- ✅ No missing translation warnings in console

---

## Known Issues and Limitations

### Phase 6 Scope

**Not Included in Phase 6:**
1. User profile editing
2. Password change functionality
3. Session expiration warnings
4. Multiple device session management
5. Admin user management UI

**Planned for Future Phases:**
- Phase 7: User management and board sharing UI
- Phase 8: Enhanced session management
- Phase 9: Audit logging UI

---

## Troubleshooting

### Issue: UserMenu Not Appearing

**Possible Causes:**
1. User not authenticated
2. Frontend not connected to backend
3. Auth token invalid or expired

**Resolution:**
- Check browser console for errors
- Verify backend is running
- Try logging out and back in

---

### Issue: Model Manager Tab Still Visible for Non-Admin

**Possible Causes:**
1. User object not loaded in Redux state
2. Cached admin status from previous session
3. Frontend code not updated

**Resolution:**
- Hard refresh (Ctrl+Shift+R)
- Clear browser cache and localStorage
- Verify `user.is_admin` value in Redux DevTools

---

### Issue: Logout Doesn't Redirect to Login

**Possible Causes:**
1. React Router not properly configured
2. Navigation hooks not working
3. JavaScript error preventing navigation

**Resolution:**
- Check browser console for errors
- Verify React Router is installed and configured
- Check that `useNavigate` hook is working

---

## Success Criteria

Phase 6 is considered complete when:

- ✅ UserMenu component displays correctly for all users
- ✅ Admin badge shows only for administrator users
- ✅ Logout functionality works reliably
- ✅ Model Manager tab is hidden for non-admin users
- ✅ Model Manager tab shows access denied for non-admin direct access
- ✅ All linters pass without errors
- ✅ Frontend builds successfully
- ✅ No visual regressions in existing UI
- ✅ All manual tests pass
- ✅ No critical accessibility issues

---

## Documentation

### Phase 6 Summary Document

Create `phase6_verification.md` with:
- Implementation checklist
- Code quality assessment
- Test results summary
- Known limitations
- Integration points with other phases

---

## Next Steps

After Phase 6 completion:
1. Begin Phase 7: User Management UI (Admin panel)
2. Implement board sharing UI
3. Add user profile editing
4. Enhance session management

---

## Appendix: API Endpoints Used

### Logout
```
POST /api/v1/auth/logout
Authorization: Bearer <token>

Response:
{
  "success": true
}
```

### Get Current User
```
GET /api/v1/auth/me
Authorization: Bearer <token>

Response:
{
  "user_id": "...",
  "email": "...",
  "display_name": "...",
  "is_admin": true/false,
  "is_active": true/false
}
```

---

*Document Version: 1.0*  
*Last Updated: January 12, 2026*  
*Phase: 6 (Frontend UI Updates)*
