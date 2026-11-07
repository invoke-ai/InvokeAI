# Models catalog

Paths are relative to the webv2 `src/` directory.

`features/models/data/modelsStore.ts` owns the shared catalog snapshot and by-key index. List refreshes are single-flight with at most one queued trailing read. Local model replacement, narrow patches (including optimistic rollback), deletion, and cover changes advance a mutation generation. A read started before that generation cannot replace the catalog or publish an obsolete error; it queues an authoritative trailing read so unrelated server changes remain visible. Cover URL versions remain local cache-busting state. Only successful mutation callers apply confirmed changes; a failed request alone does not supersede a valid read.

Account reset clears the catalog and flight state. Responses from an earlier account lifetime cannot update the current catalog or schedule reads for it.
