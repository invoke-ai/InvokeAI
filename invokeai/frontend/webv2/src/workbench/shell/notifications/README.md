# Notifications

Paths are relative to the webv2 `src/` directory.

Workbench notifications are plain data in workbench state (`WorkbenchNotification` in `workbench/projectContracts.ts`). They are session-only: hydration empties the list and the saved session does not include it. Workbench state must still stay plain, cloneable data, so a notification never holds a closure. Callers record through `useNotify` (`workbench/useNotify.ts`) or `commands.notifications.add`; `WorkbenchNotificationToaster` toasts each one recorded while the editor is mounted, subject to `toastPolicy.ts`. Outside the editor `useNotify` falls back to plain platform toasts.

## Actions

A notification offers buttons through `actions`, a list of `WorkbenchNotificationAction` descriptors such as `{ kind: 'open-widget', widgetId: 'gallery' }`. `notificationActions.ts` maps every kind to a handler: `label` returns the translated button text, or null when this editor cannot offer the action (the widget is not enabled, the undo is gone or its project closed), and `run` acts through the `NotificationActionContext`; optional `hold` and `release` bracket the time a toast shows the action. `editorNotificationActions.ts` builds that context from the editor's store, widget registry and focus: `open-widget` brings a placed widget forward (raising a floating window) or opens it, then moves focus there. A new kind is a union member plus a handler; the handler table's mapped type fails to compile until both exist. Handlers that need another owner's command reach it through that owner's public entry or a context field, never through the descriptor.

Toasts with actions stay for `ACTION_TOAST_DURATION_MS` (hovering or focusing pauses them) and close when the editor unmounts, so no button outlives the store it acts on. `persistent: true` keeps a toast until the user closes it; use it for failures the user must not miss. Persistent toasts also close with the editor, or outside it with the account, so they never carry over to the next user. A repeat of the newest error coalesces into it and takes the repeat's actions, which act on the latest occurrence. Only a repeated run outcome toasts again; an ambient repeat updates the record while a toast still showing keeps its old buttons.

## Undo

`notify.undoable(title, { message, onUndo, projectId })` reports an action that already happened, the house style for reversible changes. It records a success notice carrying `{ kind: 'undo', undoId, projectId }`; the closure waits in `undoRegistry.ts`, a session-only map cleared on account change. Undo runs at most once and only while its toast is open: the toast holds the entry, and closing or timing out the toast releases it. Entries no toast has shown are capped at 20, oldest dropped first; a shown entry is never dropped, so a visible Undo always works. Outside the editor there is no center: `undoable` shows its Undo toast directly, closed with the account.

The toast lives as long as the editor, not the project tab. `onUndo` therefore captures the ids of what it reverts, never "the active project", and does nothing (or reports) when its target is gone. With `projectId`, the toast withdraws Undo, and a press does nothing, once that project is closed.

Confirmation dialogs are for permanent or bulk actions, not a substitute for undo.

## Platform toaster

`createActionToast` (`platform/ui/toaster.tsx`) renders buttons, primary first; each dismisses the toast before it acts. A `signal` dismisses the toast when its scope ends, and an already-ended signal shows nothing. The toast region takes focus with Alt+T, and Tab then reaches each toast and its buttons. The close button's name comes from `common.dismissNotification`.
