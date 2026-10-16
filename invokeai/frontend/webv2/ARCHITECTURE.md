# Webv2 module architecture

Four top-level module groups:

- `app/`: providers, routing, boot, composition roots.
- `platform/`: application infrastructure. No workbench or feature dependency.
- `workbench/`: atomic project aggregate, layout, shell, widget runtime, extension runtime.
- `features/<domain>/`: domain behavior behind a few public top-level entry modules (`features/<domain>/index.ts` first).

Paths are relative to `src/`, except paths starting with `src/`, `scripts/`, `performance/`, or `artifacts/`: those are relative to this package. Workbench ownership tables use paths relative to `src/workbench/`.

Dependency direction: `app -> workbench/features -> platform`.

- Feature callers import only a feature's public entry modules: the top-level modules registered in `src/architecture/featureInterfaces.ts`. `index` is implicitly public. Unregistered feature or module is private (fail-closed).
- Everything under `core/`, `data/`, `ui/`, and runtime implementation is private.
- Features may group substantial behavior under `core/`, `data/`, `ui/`. Pure cores never import transport, Data, React, or UI modules, type-only imports included.
- Production canvas callers use `workbench/canvas-engine/api.ts`, not the engine implementation root.

## App and Platform ownership

Ownership follows invariants, not the first caller:

| Module                 | Owned implementation                                                                                                                                  | Must not own                                                    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `app/`                 | `main`, provider assembly, routes and guards, route runtimes, adapter selection, cross-module dependency construction, theme/language DOM controllers | Domain behavior, transport implementation, reusable state or UI |
| `platform/transport`   | HTTP primitives and URL resolution, injected auth hook, Socket.IO mechanics, connection snapshots                                                     | Auth token policy, Queue/model/node event interpretation        |
| `platform/state`       | React-free listener/store/selector cores and separate React hook adapters                                                                             | Domain stores or aggregate state                                |
| `platform/query`       | Query client construction and defaults                                                                                                                | Feature query keys, DTOs, invalidation policy                   |
| `platform/i18n`        | i18next bootstrap and supported-language metadata                                                                                                     | Persisted preference ownership                                  |
| `platform/logging`     | Event contracts, recording filters, safe normalization, bounded in-memory history, subscriptions, console output, export envelope                     | Preference persistence, diagnostic presentation, event policy   |
| `platform/ui`          | Chakra system/theme, reusable controls, dialog root and modal presence, toaster, brand mark                                                           | Workbench widgets or feature-specific views                     |
| `platform/runtime`     | Domain-neutral build/runtime metadata                                                                                                                 | Route or feature policy                                         |
| `workbench/components` | Workbench hydration and Queue progress presentation                                                                                                   | Reusable control or theme implementation                        |

- `src/` root holds declaration files only. Runtime boot files live in App. Old root entry paths are gone; `index.html` names `src/app/main.tsx` directly.
- `@theme` resolves to Platform UI theme implementation. Every other Platform caller uses `@platform/*`. Moved App, HTTP, socket, external-store, selector, theme, i18n, reusable-UI modules have no compatibility path.
- Authentication: `app/main.tsx` hands it to Platform transport before React mounts. Identity owns the credential and every policy about it. Platform captures it once per request and reports that request's outcome.
- Platform socket publishes raw events and connection transitions. Queue and other domains interpret them.
- Lifecycle-specific production composition stays explicit at its natural root: authenticated socket in `app/router.tsx`, editor runtimes in `app/WorkbenchApp.tsx`, feature UI providers in `app/workbenchPorts.tsx`, lazy editor bridges beside their lazy entry modules. App builds the production Queue runtime from Queue, Workbench, Gallery, Workflow adapters.
- Platform State owns reusable projections of broad external stores into referentially stable read-only stores. App may build them for a feature port. Listener, caching, equality mechanics stay domain-neutral Platform code.

`pnpm architecture:check` is the executable source of truth for dependency direction. It parses TypeScript and classifies static imports, type imports, dynamic imports, re-exports, inline imported types, after resolving aliases and relative paths. Temporary violations go in `src/architecture/migrationExceptions.ts` with owner and removal ticket. Architecture is complete only when that registry is empty.

## Owner documents

Behavior specs live beside their owning code. Read the owner's document before changing its behavior.

| Topic                                                         | Document                                                                               |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Identity credential and session lifetimes                     | [features/identity/README.md](src/features/identity/README.md)                         |
| Application and project event logging                         | [platform/logging/README.md](src/platform/logging/README.md)                           |
| Queue backend port and active sessions                        | [features/queue/README.md](src/features/queue/README.md)                               |
| Gallery listings and reveal                                   | [features/gallery/README.md](src/features/gallery/README.md)                           |
| Intermediates manager and cleanup                             | [features/intermediates/README.md](src/features/intermediates/README.md)               |
| Font catalog and registration                                 | [features/fonts/README.md](src/features/fonts/README.md)                               |
| Project documents, workflows, persistence, and recovery       | [workbench/projects/README.md](src/workbench/projects/README.md)                       |
| Widget placement, floating windows, focus, and shortcut hints | [workbench/widget-frame/README.md](src/workbench/widget-frame/README.md)               |
| Settings contributions and the settings dialog                | [workbench/settings/README.md](src/workbench/settings/README.md)                       |
| Widget manifests and extension discovery                      | [workbench/extensions/README.md](src/workbench/extensions/README.md)                   |
| Canvas engine                                                 | [workbench/canvas-engine/ARCHITECTURE.md](src/workbench/canvas-engine/ARCHITECTURE.md) |
| Performance gates and byte budgets                            | [performance/README.md](performance/README.md)                                         |

## Contract ownership

- Contracts live with the module owning their invariants. Retired `workbench/types.ts` path is an architecture violation. Workbench aggregate, widget, Canvas, persistence, settings, feature contracts each have an owner-local canonical path.
- Persisted schemas and backend DTOs stay private versioned/wire values. They cross seams only through explicit serializers, hydrators, mappers.
- `platform/core/outputImages.ts` is the explicit shared adapter for backend invocation-output image names. Platform owns it because Queue and Nodes both need the projection and cannot import each other's private code. It recognizes only direct image, image-collection, workflow-return output contracts. Never grow it into a recursive walker of arbitrary metadata or input fields.
- Imports of another feature's `core/`, `data/`, `ui/`, or runtime implementation fail outside a ticketed migration exception.
- Wildcard exports may aggregate files inside one owner. Never across an ownership seam.
- `pnpm architecture:check` generates the contract caller matrix; `pnpm test:performance:build` generates development-invalidation budgets.
- Compatibility exports under `workbench/` are temporary. New behavior goes behind the target module interface. Compatibility files gain no new implementation logic.

## Workbench ownership destinations

`src/architecture/workbenchOwnershipManifest.json` is the executable current-to-target ownership manifest. `pnpm architecture:check` proves every production Workbench module has exactly one target owner. It writes the import/export inventory to `artifacts/architecture/workbench-ownership-inventory.json`, regenerated as modules change; tests and type-test fixtures excluded on purpose.

Every manifest directory rule targets Workbench. One override left: `workbench/components/QueueProgressIndicator.tsx` targets Queue, due to move to `features/queue/ui/`. Identity, Models, Nodes, Queue, Gallery, Generation, Workflow, Upscale code lives under `features/`. Domain-neutral React adapters `useMountEffect` and `useModifierHeld` live in `platform/react/`, with no Workbench import.

| Workbench path                                                                                       | Locality decision                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| aggregate root modules (including persistence), `projects`, `settings`                               | Atomic project/account state and its persistence policy remain local to the aggregate. Persistence mechanics move behind the aggregate's adapter. Persisted project queue history stays in Workbench. |
| `shell`, `components/WorkbenchSplashScreen`, `launchpad` shell, `widget-frame`, non-domain `widgets` | Route-independent shell, layout, widget hosts, and registration are Workbench implementation.                                                                                                         |
| `extensions`, `hotkeys`, `diagnostics`                                                               | These enforce the Workbench extension runtime; they are not independent vertical domains.                                                                                                             |
| `canvas-engine`, `canvas-operations`, Canvas/layers widgets, root Canvas modules                     | Canvas is a deep internal Workbench module behind `canvas-engine/api.ts`, not another top-level Feature.                                                                                              |
| `image-actions`                                                                                      | Cross-surface image orchestration and transient display stay internal until a second independent implementation justifies a seam.                                                                     |

- Overrides are longest-prefix rules: a mixed folder's split is explicit, not inferred from imports.
- Compatibility re-exports keep the target owner of what they forward. They are deletion work, never a fifth owner.
- Owner dependency graph is acyclic. Only direction: `app -> workbench/features -> platform`. Cross-owner access only through public module interfaces or composition-root adapters. `src/architecture/workbenchOwnership.test.ts` fails on any strongly connected component among owners. `src/architecture/migrationExceptions.ts` is empty.

Feature interfaces stay ordinary module interfaces unless two adapters add leverage.

- Queue backend port: production transport/realtime adapter plus test adapter.
- Identity token/session port: browser-session adapter plus test adapter.
- Models and Nodes: production event/transport adapters. Their sinks (model-load activity, node execution) are injected into Queue's runtime at the App composition root; Queue runtime tests use doubles.
- Gallery, Generation, Workflow, Upscale keep compilation and UI seams internal until a feature ticket shows another independent adapter. Workbench image actions also stay internal, not a shallow Image Feature.
- Gallery picker and media slots read a narrow `GalleryHost`: gallery scope, notifications, optional Gallery-widget reveal. Workbench derives it from the Gallery UI adapter. Launchpad's model manager has no workbench and supplies its own.
- Media slots accept dragged gallery items only inside `GalleryDragScope`, provided by the workbench shell with its drag context. Widget dialog hosts and the Launchpad sit outside it.

| Feature       | Responsibility and representative callers                                                                                                                                | Allowed dependencies and candidate public interface                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Identity      | Session/token policy and user administration; App guards/login, Queue capabilities, Workbench account/Launchpad surfaces                                                 | Platform plus Models' public capability reads; session reads/commands, token port, user queries and manager contribution                                                                               |
| Models        | Catalog, configuration, install/load lifecycle; Generation, Queue, Workflow, Upscale, Identity, model manager                                                            | Platform plus Identity's public authorization reads and Gallery's public picker; descriptors/capabilities, catalog/load commands and manager contribution                                              |
| Nodes         | Custom-node catalog/templates and execution lifecycle; Workflow editor and node manager                                                                                  | Platform; descriptors, catalog commands, execution snapshots and manager contribution                                                                                                                  |
| Queue         | Backend queue state, realtime interpretation, submission/reconciliation and Queue UI; App runtime, Workbench aggregate/shell, Generation/Gallery/Workflow                | Platform and public envelopes/capabilities from Identity, Models, Generation, Workflow, Upscale; backend port, queue reads/commands and widget contribution                                            |
| Gallery       | Image collection, selection, board policy and Gallery UI; App adapter, Generation, Workflow, Upscale, Models, Workbench image actions                                    | Platform and Queue's public progress reads; gallery reads/commands and widget contribution                                                                                                             |
| Generation    | Generation settings, validation, graph compilation and Generate UI; Workbench invocation, Queue, Gallery, Workflow, Upscale                                              | Platform and public Models/Gallery/Queue interfaces; settings/validation, compiled submission envelope and widget contribution                                                                         |
| Workflow      | Workflow document, validation, compilation, preview/editor/library and explicit publication; App adapter, Workbench invocation, Nodes                                    | Platform and public Nodes/Models/Gallery/Queue/Generation interfaces; document reads/commands, compiled submission envelope and widget contribution                                                    |
| Upscale       | Upscale settings, validation, compilation and UI; Workbench invocation and Queue                                                                                         | Platform and public Models/Gallery/Queue interfaces; value/validation reads, submission command and widget contribution                                                                                |
| Intermediates | Per-project intermediates read models, cleanup previews/operations and the Settings section that manages them; Workbench settings, project menus, account administration | Platform plus Identity's public session/capability reads; query keys, read models, the focus intent, the browser hold lease and the `settingsContribution` metadata whose loader owns the section body |
