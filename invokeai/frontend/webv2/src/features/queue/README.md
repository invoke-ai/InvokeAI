# Queue

Paths are relative to the webv2 `src/` directory.

Queue's private `QueueBackendPort` is adapter seam; public callers use domain reads, commands, and production runtime factories from `@features/queue`. Private snake_case HTTP DTOs map explicitly to camelCase Queue read models before crossing that interface. TanStack Query owns global backend queue read model (reads and hydrates only a bounded recent window) and a status-only read for count summaries such as Home; non-React runtimes own reconciliation and realtime orchestration. Persisted per-project queue history stays in Workbench aggregate.

Queue exposes board-independent active backend sessions, including preparation and result settlement. Gallery progress section includes every unfinished batch slot, queued work too; Preview follows executing and settling sessions. Both use gallery-destination projections of active project's queue; eligibility is presentation policy, separate from discovery. Workbench owns transient overview/pinned follow selection for its mounted lifetime, clears pins on project changes or when following stops, and never persists pins or preview frames. Live sessions take no part in board counts, saved-image selection, or navigation. Saved outputs keep their actual backend board destinations; automatic result selection does not change browsed board or its paging window.
