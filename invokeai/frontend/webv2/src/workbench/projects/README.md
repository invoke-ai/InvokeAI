# Project persistence

The backend project record is authoritative. Writes use `expected_revision`; a divergent revision or remotely deleted project requires an explicit user decision. Editing can continue while that decision is pending. Saving a copy reuses one reserved identity through `createProjectSettled`, including after a lost response.

`ProjectDocumentV3` allowlists editable document fields. Queue runs, events, undo and per-workflow edit histories are not project documents. Documents are limited to 32 MiB of UTF-8 JSON on both sides of the API.

## Workflows

A project owns an ordered workflow collection (`workflows.entries`) and remembers its active workflow. Each entry carries the editable document, optional source metadata (the library template it was opened from or last explicitly saved to, with the content revision then observed, or `null` when unknown) and the newest successful run's output. Source metadata sits beside the document, outside graph edit history, so undoing an edit never undoes a save. `projectWorkflows.ts` owns membership, selection and the session histories (one per workflow, 40 entries per project across all of them, structurally shared); Workflow owns editing, serialization, library transport and presentation.

Projects autosave their workflows. The library changes only through explicit publication (`features/workflow/data/publication.ts`): saving as a new template or updating the source at exactly the revision the copy last saw; a stale revision is a 409 the user resolves. Every media reference in every workflow, active or not, takes part in reference indexing, cleanup protection, duplication and archive export.

`migrateProjectDocument` is the one migration boundary. Unversioned (schema 1) and schema 2 documents become schema 3 with their single `projectGraph` as the first, active workflow; a `libraryWorkflowId` becomes a source with an unknown revision, never assumed to match the library's current contents. Server loads, recovery drafts, duplication and `.invk` import all pass through it; every write path emits schema 3. Malformed collections (dangling active id, repeated ids, damaged entries) and structurally damaged workflow documents are refused rather than repaired or replaced with blanks; a refused document is reported, left untouched on the server, and its raw content stays available for export. Same-server duplication keeps source references; `.invk` import clears them, because a portable file never names a library write target.

Canvas uploads carrying a project id await that identity’s acknowledged server creation through the mounted persistence service. Uploads waiting on an unacknowledged project share one serialized creation attempt, and a failed attempt answers further uploads for a few seconds before another is made; later uploads use the acknowledged identity without flushing unrelated edits. Only a closed project, an account change, engine disposal or the operation's own cancellation stops an upload, and those stop the wait immediately. Any other outcome (creation failed or offline, a conflict, or a server that no longer knows the project and refuses the id) sends the upload without a project id. Such uploads are never re-associated with the project later, so the project's intermediates cleanup does not see them.

Project links are consumed after a successful open, so reloading restores the saved active project rather than replaying an earlier link. A new-project startup selects the newly created draft before saving the session.

## Browser recovery

- Account-owned IndexedDB stores only unacknowledged project drafts, active queue runs, receipt acknowledgements, and bounded recall values. Clean server documents are not mirrored.
- Draft generations fence acknowledgements. Draft writer ownership and cross-tab notifications prevent one editor from overwriting another editor's unacknowledged work.
- Queue ownership uses Web Locks. Journal writes precede submission; backend receipts make retries safe after lost responses. Accepted IDs must be durable before their receipt can expire. Fresh runs that never reached browser storage use best-effort terminal acknowledgement instead.
- Completed, failed, and cancelled runs retain exact recall values in an LRU cache, capped at 500 entries / 32 MiB. Eviction removes recall convenience, not project data or active-run recovery.
- Projects with journal entries remain reachable even if another tab saved an empty session. Automatic reopening is bounded; remaining entries can be opened, exported as queue-recovery JSON, or explicitly discarded. Discarding local recovery does not cancel backend work.
- Browser-storage failures do not block backend project loads or saves. Recovery warnings distinguish degraded local durability from successful server persistence.

## Intentional cutover

The old workbench mirror, sync map, and refused-project localStorage keys are deleted on initialization. Their contents are not migrated. Export any needed browser-only work with the old build before upgrading.

Cold-start offline editing is not supported. When the backend cannot load, the unavailable screen provides retry and local draft/run exports. Conflicted or schema-refused drafts are retained until explicitly resolved or deleted; they are never silently evicted.

## Font dependencies

Custom text retains an immutable font reference and explicit variation coordinates. Browser registration names are runtime state and never enter the project document. New typography requires Canvas schema 4; documents without it retain their existing compatibility floor.

Project archives record font dependencies by content hash. Export defaults to references only; **Include font files** embeds the original referenced files, once per hash, after authenticated download and checksum verification. A failed required font download fails the embedded export rather than producing a silently incomplete archive.

Import verifies declared entries and checksums, then validates all embedded fonts before creating server resources. Fonts are uploaded privately and the canonical document's references are remapped to the destination IDs. Duplicate uploads reuse existing private records. Rollback removes only newly created fonts and only when project creation has not been attempted or has confirmed absence; an ambiguous create response retains the resources. Font quota failures offer an explicit references-only retry. Changing accounts cancels the operation and prevents cleanup under another user's credentials.

## Verification

Run `pnpm lint`, `pnpm test`, `pnpm test:browser`, and `pnpm run test:performance:build` from webv2. Queue receipt tests also cover backend admission, account isolation, project deletion, partial acceptance, and idempotent retries. Browser tests exercise actual IndexedDB transactions and Web Locks.
