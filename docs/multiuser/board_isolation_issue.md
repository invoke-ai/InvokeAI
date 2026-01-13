# Board Isolation Issues in Multiuser Implementation

## Problem Description

After implementing multiuser support (Phases 1-6), there are several board isolation issues that need to be addressed:

### 1. Board List Not Updating When Switching Users
**Issue:** In the Web UI, when a user logs out and logs back in as a different user, the board list is not updated unless the window is refreshed.

**Expected Behavior:** Board list should automatically update to show only the new user's boards when switching users.

**Current Behavior:** Old user's boards remain visible until manual page refresh.

**Root Cause:** Frontend Redux state is not being cleared on logout, leading to stale board data.

### 2. "Uncategorized" Board Shared Among Users
**Issue:** The default "Uncategorized" board appears to be shared among all users instead of being user-specific.

**Expected Behavior:** Each user should have their own isolated "Uncategorized" board for images not assigned to any board.

**Current Behavior:** All users see and share the same "Uncategorized" board.

**Root Cause:** The special "none" board_id (representing uncategorized images) is not being filtered by user_id in queries.

### 3. Admin Cannot Access All Users' Boards
**Issue:** Administrator users should be able to view and manage all users' boards, but currently cannot.

**Expected Behavior:** 
- Admin users should see all boards from all users
- Board names should be labeled with the owner's username for clarity (e.g., "Floral Images (Lincoln Stein)")
- Admin should have appropriate permissions to manage boards

**Current Behavior:** Admin users only see their own boards like regular users.

**Root Cause:** Board queries filter by current user's user_id without special handling for admin role.

## Technical Details

### Database Schema
The migration_25 already adds:
- `user_id` column to `boards` table with default 'system'
- `is_public` column to `boards` table
- `shared_boards` table for board sharing
- Indexes on user_id and is_public

### Areas Requiring Changes

#### Backend (Python)
1. **Board Records Service** (`invokeai/app/services/board_records/`)
   - Update queries to handle admin users specially
   - Ensure proper user_id filtering for regular users
   - Handle "uncategorized" (none board_id) per-user isolation

2. **Board Service** (`invokeai/app/services/boards/`)
   - Add admin check in `get_many()` method
   - Update board DTOs to include owner information for admin view
   - Ensure all board operations respect user ownership

3. **API Endpoints** (`invokeai/app/api/routers/boards.py`)
   - Update endpoints to check for admin role
   - Add owner username to board responses for admin users
   - Ensure proper authorization checks

#### Frontend (TypeScript/React)
1. **Redux State** (`invokeai/frontend/web/src/features/gallery/store/`)
   - Clear board state on logout
   - Refresh board list on login
   - Handle board ownership display

2. **Board Components**
   - Update board display to show owner for admin users
   - Add visual indicators for owned vs. other users' boards
   - Update board selection logic

3. **Auth Flow**
   - Ensure state cleanup on logout
   - Trigger board list refresh after login

## Implementation Plan

### Phase 1: Backend Board Isolation
1. Update board record queries to filter by user_id (except for admins)
2. Add admin role check to bypass user_id filtering
3. Handle "uncategorized" board per-user isolation
4. Add owner information to board DTOs for admin users

### Phase 2: Frontend State Management
1. Add logout action to clear all board state
2. Add login success action to refresh board list
3. Update board selectors to handle admin view

### Phase 3: UI Updates
1. Display owner username for admin users
2. Add visual distinction between own and others' boards
3. Update board creation/management permissions

### Phase 4: Testing
1. Test board isolation for regular users
2. Test admin can see all boards
3. Test uncategorized board per-user isolation
4. Test state cleanup on logout/login
5. Test board sharing functionality

## Acceptance Criteria

- [ ] Regular users only see their own boards and shared boards
- [ ] Each user has their own "Uncategorized" board
- [ ] Admin users see all boards from all users
- [ ] Board names show owner for admin view (e.g., "Board Name (Username)")
- [ ] Logging out and logging in as different user updates board list immediately
- [ ] No stale board data persists after user switch
- [ ] Board sharing works correctly
- [ ] All board operations respect user ownership
- [ ] Tests validate board isolation for all scenarios

## Related Files

### Backend
- `invokeai/app/services/board_records/board_records_sqlite.py`
- `invokeai/app/services/board_records/board_records_base.py`
- `invokeai/app/services/boards/boards_default.py`
- `invokeai/app/services/boards/boards_base.py`
- `invokeai/app/api/routers/boards.py`

### Frontend
- `invokeai/frontend/web/src/features/gallery/store/gallerySlice.ts`
- `invokeai/frontend/web/src/features/gallery/components/Boards/BoardsList.tsx`
- `invokeai/frontend/web/src/features/auth/store/authSlice.ts`

### Tests
- `tests/app/services/boards/test_boards_multiuser.py` (needs expansion)
- Frontend tests (to be added)

## Priority
**High** - These issues affect the core multiuser functionality and user experience.

## Dependencies
- Phases 1-6 of multiuser implementation (completed)
- Migration 25 (completed)

## Recommended Approach

Create a new GitHub issue with title: `[enhancement]: Fix board isolation in multiuser implementation`

Then create a new PR that addresses all three issues together since they are closely related and affect the same subsystems (board service, API, and frontend state management).
