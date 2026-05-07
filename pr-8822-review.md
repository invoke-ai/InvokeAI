# Code Review: PR #8822 - Multi-User Mode

**PR:** https://github.com/invoke-ai/InvokeAI/pull/8822
**Author:** lstein
**Head SHA:** `b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3`
**Scope:** 118 files changed, +12,200 / -380 lines

---

## Summary

This PR adds multiuser support to InvokeAI, allowing multiple users to share a single backend with isolated boards, images, assets, and queue items. Includes user authentication (JWT), role-based access control (admin vs regular user), WebSocket event filtering, database migrations, and a full frontend auth flow (login, admin setup, user menu).

---

## Issues Found

### High Confidence (score >= 80)

**1. `boards.get_all()` missing required `is_admin` argument -- will crash at runtime** (score: 100)

In `invokeai/app/services/shared/invocation_context.py`, the call to `boards.get_all()` is missing the required `is_admin` positional argument. The method signature was changed to `get_all(self, user_id, is_admin, order_by, direction, include_archived)` but this call site only passes `user_id`, `order_by`, and `direction`. This will cause a `TypeError` whenever any invocation node calls `context.boards.get_all()`.

https://github.com/invoke-ai/InvokeAI/blob/b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3/invokeai/app/services/shared/invocation_context.py#L103-L106

---

**2. Token verification accepts tokens without `exp` field indefinitely** (score: 100)

In `invokeai/app/services/auth/token_service.py`, the expiry check is `if "exp" in payload:` -- if `exp` is absent, the token is accepted with no expiration. While server-generated tokens always include `exp`, a crafted token (by anyone with knowledge of the JWT secret) would never expire. The check should require `exp` to be present and reject tokens without it.

https://github.com/invoke-ai/InvokeAI/blob/b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3/invokeai/app/services/auth/token_service.py#L91-L99

---

**3. `clear` endpoint wipes ALL users' queue items, not just the caller's** (score: 100)

The `clear` endpoint in `session_queue.py` calls `session_queue.clear(queue_id)` which deletes ALL queue items for ALL users. The endpoint only checks ownership of the currently-executing item, but then unconditionally wipes the entire queue. A non-admin user can clear every other user's pending jobs. The underlying `SqliteSessionQueue.clear()` method has no `user_id` parameter, unlike `prune` and other similar methods that were updated for multi-tenancy.

https://github.com/invoke-ai/InvokeAI/blob/b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3/invokeai/app/api/routers/session_queue.py#L327-L346

---

### Moderate Confidence (score 50-79)

**4. `resume`/`pause` endpoints use `AdminUser` instead of `AdminUserOrDefault` -- broken in single-user mode** (score: 75)

The `resume` and `pause` endpoints use `AdminUser` which requires an authenticated token. In single-user mode (default deployment), no tokens exist, so these endpoints return 401. Other admin-only endpoints correctly use `AdminUserOrDefault` which falls back to a system admin in single-user mode.

https://github.com/invoke-ai/InvokeAI/blob/b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3/invokeai/app/api/routers/session_queue.py

---

**5. Board `update_board`/`delete_board` have no ownership check** (score: 75)

Both endpoints inject `current_user` but never verify the board belongs to that user. The docstrings say "user must have access to it" but no validation occurs. Any authenticated user can modify or delete any other user's boards by board ID.

https://github.com/invoke-ai/InvokeAI/blob/b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3/invokeai/app/api/routers/boards.py

---

**7. `_handle_unsub_queue` doesn't leave user/admin rooms** (score: 75)

`_handle_sub_queue` joins three rooms (queue, `user:{user_id}`, and `admin`), but `_handle_unsub_queue` only leaves the queue room. Unsubscribed sockets remain in user/admin rooms and continue receiving private events.

https://github.com/invoke-ai/InvokeAI/blob/b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3/invokeai/app/api/sockets.py

---

**8. Direct image fetch endpoints have no authentication** (score: 75)

`get_image_full`, `get_image_dto`, `get_image_thumbnail`, and `get_image_metadata` have no `current_user` dependency. Any user who knows an `image_name` can fetch any other user's images directly, bypassing the user isolation that was added to list endpoints.

https://github.com/invoke-ai/InvokeAI/blob/b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3/invokeai/app/api/routers/images.py

---

**9. SocketIO `cors_allowed_origins="*"` while HTTP CORS is configurable** (score: 75)

The Socket.IO server hardcodes `cors_allowed_origins="*"` while the HTTP layer uses configurable `allow_origins` from `app_config`. Any webpage can connect to the WebSocket even if HTTP CORS is restricted, which is a security regression now that token-based socket auth exists.

https://github.com/invoke-ai/InvokeAI/blob/b4ab173b3c7a1d4551b6be0fe45af9af9aeb8bd3/invokeai/app/api/sockets.py

---

### Lower Confidence (score < 50)

**11. `authenticate()` updates `last_login_at` for disabled users** (score: 25)

The `is_active` check happens in the router after `authenticate()` returns, so `last_login_at` is updated even for deactivated accounts.

**12. `get_queue_item_ids` endpoint has no authentication** (score: 50)

Returns item IDs without requiring auth, though the exposure is limited since it only returns IDs, not full items.

**13-14. Missing colocated tests for `validatePasswordStrength` and `getBadgeText`** (score: 50)

Pure functions with non-trivial branching logic added without `.test.ts` files. CLAUDE.md requires colocated tests for non-trivial logic, but these are borderline given "We currently do not do UI tests."

---

### False Positives Identified

- **`sanitize_queue_item_for_user` shallow copy**: The shallow copy is safe here because all fields are either set to `None` or replaced with new objects, never mutated in place.
- **Migration ordering conflict**: Already acknowledged by lstein in PR comments and consolidated into a single migration file.
- **Redundant `hasattr` guards in `_handle_queue_event`**: While technically always true (since `user_id` has a default), this is defensive coding and doesn't cause bugs.
- **`list_image_dtos` missing `is_admin` parameter**: Verified manually -- admin can see all users' images in practice, so the filtering works correctly despite the parameter not being explicitly passed.
- **`list_queue_items` missing user JOIN**: `list_queue_items` is never called by any router or service -- only `list_all_queue_items` (which has the correct JOIN) is actually used.
