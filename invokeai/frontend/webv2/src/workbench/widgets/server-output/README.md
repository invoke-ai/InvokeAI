# Server Output

Paths are relative to the webv2 `src/` directory.

Bottom-region widget that shows the server's Python console output, read over `GET /api/v1/app/server_output`. The project keeps only placement. The transcript lives in memory while the expanded view is mounted. Line wrapping is an account preference (`serverOutputWrapLines`, on by default), edited from the widget's settings cog and the Settings dialog.

## Availability

- The manifest declares `requiresCapability: 'viewServerOutput'` ([extensions](../../extensions/README.md#availability)). The grant needs both Identity's `canViewServerOutput` (single-user mode, or a multi-user admin) and `can_view_server_output` from the account's frontend config (`workbench/frontendConfig.ts`). An unanswered config is pending, and so is one being retried. A failed or older config (no field) denies.
- `useWidgetCapabilities()` observes the config query itself, only while the session's role could be granted. The status bar, the Widgets dialog and this view use it. While the status bar is mounted, a single-user or admin editor reads the config once at startup; an ordinary multi-user account never does. A new observer retries a failed read, so opening the view is a retry.
- A saved layout restores the widget by ID without discovery, so the expanded view checks the grant before mounting the reader; a refused widget never reads the transcript. The refusal names its reason:
  - forbidden: the session is not an admin, or the server answered 401/403;
  - unavailable: single-user mode with capture off, or the server answered 404;
  - denied: the config refuses a multi-user admin session without saying why.
- Discovery hides a refused widget, chip included, so the refusal offers **Remove from layout** (the shell's undoable close) and, when the role could be granted, **Check again**.
- A refused read clears the rows, stops the reader and invalidates the config. Its reason stays on screen even after the config answers, so the message never changes under the reader. **Check again** reads the config; if it grants, a fresh reader mounts. Collapsing and expanding mounts a fresh view, which reads the grant again too.

## Reader

`reader.ts` is a non-React runtime that the expanded view creates and starts with `useMountEffect`. Collapsing the bottom panel unmounts the view, which stops the reader. Hidden React Activity stops it too; revealing it resumes from the cursor.

- At most one GET is in flight. The next is scheduled after it settles:
  - at once while `has_more`;
  - 250 ms after a read that changed something;
  - 500 ms, doubling to 2 s, while reads come back unchanged.
- A transient failure (network error, 5xx, anything unrecognized) shows Reconnecting and keeps the rows. It retries from the last applied cursor after 1 s, doubling to 15 s; a successful read resets that backoff.
- A 422 drops the cursor and reads a fresh snapshot at once. A 422 on a snapshot read counts as transient.
- Stopping, hiding the document, starting again and the end of the account lifetime (`captureAccountScope()`) each take a new generation and abort the request in flight. A settlement from an older generation is dropped unapplied. When the lifetime ends, the rows clear and the reader stops for good.
- `transcript.ts` applies reads. A reset replaces everything; otherwise rows are upserted by ID and rows below `floor_id` are dropped. Rows are grouped into blocks of 64 IDs, and an unchanged block keeps its identity, so only blocks a read touched re-render. The transcript holds at most about 2 MiB (the server's own retention) and 20,000 rows; past either bound the oldest rows go, and the view says earlier output is no longer available. Each non-initial reset carries a serial, so a dismissed notice does not hide a later one.
- No transcript text goes into Query, browser storage, logging, diagnostics or telemetry. HTTP breadcrumbs carry the path without the cursor. Requests use `cache: 'no-store'`, and the service worker never handles `/api`.

## Header and controller

The view, its header label and its header action render apart. `controller.ts` joins them: the view creates a controller and registers it under its instance ID while mounted (`serverOutputControllers`), and the panel attaches its reader and copy text to it. `WidgetLabelProps.instanceId` lets the label find it.

- The label is the title followed by a dot and one word (Checking access, Connecting, Live, Reconnecting, Unavailable), from `getServerOutputStatus`. That word is the panel's only live region and changes only with the state, never with output. It shares the title's baseline, with the dot inline and centered on the word.
- The header action copies the selection inside the log, or everything held. It uses the Clipboard API and says when that is unavailable.

## View

- Rows render as text nodes. Style runs go through `rowSegments.ts`: a color is accepted only as a palette index (`0`–`255`) or a `#rrggbb`, and anything else is dropped. Text attributes become data attributes that `SERVER_OUTPUT_LOG_CSS` maps to theme tokens. ANSI 0–15 use theme tokens; the 256-color table and truecolor become inline colors.
- Blocks use `content-visibility: auto`. In local Chromium, the first render of about 10,800 rows (the 2 MiB bound) took about 270 ms. After that, a rewrite plus an append took about 2 ms to render and lay out.
- Wrapping on wraps rows in a vertical scroll area. Off keeps rows whole in a two-axis scroll area (`Scrollable orientation="both"`).
- `logViewport.ts` follows the end while the reader is there. Scrolled up, it re-pins the row at the top of the viewport after every DOM or size change, before paint, so rewrites, appends, evictions and wrap changes never move what is being read. A scroll event that arrives while the content size has changed since the last reconcile is the browser clamping, not the reader, and is ignored. Browser scroll anchoring is off on the scroller. **Jump to latest** returns to the end and focuses the log.
- The log (`role="log"`, announcements off) is focusable and shows the inside focus ring. Arrow keys and Page keys scroll it, and Ctrl/⌘+A selects the log only. A copy from the keyboard or the context menu uses the transcript's text for the selected rows, or all rows when nothing is selected; it needs no Clipboard API.
- Body states:
  - loading;
  - a failed first read ("Can't reach the server yet"; the header says Reconnecting);
  - an empty console;
  - a refusal with its actions.
- Dismissible notices say the server restarted (`reset: "restarted"`) or history expired (`reset: "expired"`). Truncated earlier output is noted at the top of the log.

## Verification

- Unit: `transcript.test.ts`, `rowSegments.test.ts`, `reader.test.ts`, and `../../widgetCapabilities.test.ts` for the availability rule and account scoping of the config query.
- Browser: `ServerOutputWidgetView.browser.test.tsx`, with the reader's delays shortened, covers:
  - discovery by grant;
  - refused restores that never read, and recovery from a failed grant read;
  - follow, and anchoring under eviction and wrap changes;
  - keyboard and partial-selection copy;
  - zero reads after close and the account switch;
  - restart, expiry and dismissal;
  - first-read failure and reconnecting;
  - 404 and a demoted admin's stable refusal and renewal.
- Mock backend: [VERIFICATION.md](../../../../VERIFICATION.md#start-from-a-known-state) lists the `/__server_output` controls.
