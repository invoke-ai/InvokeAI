# Webv2 module architecture

Webv2 uses four top-level module groups:

- `app/` owns providers, routing, boot, and composition roots.
- `platform/` owns application infrastructure with no workbench or feature dependency.
- `workbench/` owns the atomic project aggregate, layout, shell, widget runtime, and extension runtime.
- `features/<domain>/` owns domain behavior behind a small set of public top-level entry modules (`features/<domain>/index.ts` first among them).

Paths are relative to `src/` unless they start with `src/`, `scripts/`, `performance/`, or `artifacts/`, which are relative to this package. Workbench ownership tables use paths relative to `src/workbench/`.

Dependency direction is `app -> workbench/features -> platform`. Feature callers import only a feature's public entry modules — the top-level modules each feature registers in `src/architecture/featureInterfaces.ts` (`index` is implicitly public; an unregistered feature or module is private, fail-closed); everything under `core/`, `data/`, `ui/`, and runtime implementation is private. A feature may group substantial behavior under `core/`, `data/`, and `ui/`; pure cores do not import transport, Data, React, or UI modules, including through type-only imports. Production canvas callers use `workbench/canvas-engine/api.ts`, not the engine implementation root.

## App and Platform ownership

Ownership is based on invariants, not the first caller:

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

The root of `src/` contains declaration files only. Runtime boot files live in App, the old root entry paths do not exist, and `index.html` names `src/app/main.tsx` directly. `@theme` resolves to Platform UI theme implementation; all other Platform callers use `@platform/*`. No compatibility path exists for moved App, HTTP, socket, external-store, selector, theme, i18n, or reusable-UI modules.

Platform transport receives authentication through `app/main.tsx`, before React mounts: Identity owns the credential and every policy about it, while Platform captures it once per request and reports that request's outcome. The Platform socket publishes raw events and connection transitions; Queue and other domains interpret those events. Lifecycle-specific production composition remains explicit at its natural root: the authenticated socket in `app/router.tsx`, editor runtimes in `app/WorkbenchApp.tsx`, feature UI providers in `app/workbenchPorts.tsx`, and lazy editor bridges beside their lazy entry modules. App constructs the production Queue runtime from Queue, Workbench, Gallery, and Workflow adapters.

Platform State also owns reusable projections of broad external stores into referentially stable read-only stores. App may construct those projections for a feature port, but the listener, caching, and equality mechanics remain domain-neutral Platform code.

`pnpm architecture:check` is the executable source of truth for these dependency directions. It uses the TypeScript parser to classify static imports, type imports, dynamic imports, re-exports, and inline imported types after resolving aliases and relative paths. Temporary violations must be recorded in `src/architecture/migrationExceptions.ts` with an owner and a removal ticket; architecture completion requires that registry to be empty.

## Owner documents

Behavior specifications live beside the code that owns them. Read the affected owner's document before changing its behavior.

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

Contracts live with the module that owns their invariants. The retired `workbench/types.ts` path is an architecture violation; Workbench aggregate, widget, Canvas, persistence, settings, and feature contracts each have an owner-local canonical path. Persisted schemas and backend DTOs remain private versioned/wire values and cross seams only through explicit serializers, hydrators, or mappers.

`platform/core/outputImages.ts` is the explicit shared adapter for backend invocation-output image names. It is platform-owned because Queue and Nodes both need the projection and cannot import each other's private implementations. The adapter recognizes only the backend's direct image, image-collection, and workflow-return output contracts; it must not become a recursive walker for arbitrary metadata or input fields.

Feature callers import a feature's public entry modules; imports of another feature's `core/`, `data/`, `ui/`, or runtime implementation are rejected outside a ticketed migration exception. Wildcard exports may aggregate files inside one owner, but may not cross an ownership seam. The contract caller matrix and development-invalidation budgets are generated by `pnpm architecture:check` and `pnpm test:performance:build` respectively.

Compatibility exports under `workbench/` are temporary. New behavior belongs behind the target module interface and compatibility files must not acquire new implementation logic.

## Workbench ownership destinations

`src/architecture/workbenchOwnershipManifest.json` is the executable current-to-target ownership manifest. `pnpm architecture:check` proves that every production Workbench module has exactly one target owner and writes the detailed import/export inventory to `artifacts/architecture/workbench-ownership-inventory.json`. The generated inventory is regenerated as modules change; tests and type-test fixtures are intentionally excluded.

Every directory rule in the manifest targets Workbench. One override remains: `workbench/components/QueueProgressIndicator.tsx` targets Queue and is due to move to `features/queue/ui/`. Identity, Models, Nodes, Queue, Gallery, Generation, Workflow, and Upscale code lives under `features/`; the domain-neutral React adapters `useMountEffect` and `useModifierHeld` live in `platform/react/` and have no Workbench import.

| Workbench path                                                                                       | Locality decision                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| aggregate root modules (including persistence), `projects`, `settings`                               | Atomic project/account state and its persistence policy remain local to the aggregate. Persistence mechanics move behind the aggregate's adapter. Persisted project queue history stays in Workbench. |
| `shell`, `components/WorkbenchSplashScreen`, `launchpad` shell, `widget-frame`, non-domain `widgets` | Route-independent shell, layout, widget hosts, and registration are Workbench implementation.                                                                                                         |
| `extensions`, `hotkeys`, `diagnostics`                                                               | These enforce the Workbench extension runtime; they are not independent vertical domains.                                                                                                             |
| `canvas-engine`, `canvas-operations`, Canvas/layers widgets, root Canvas modules                     | Canvas is a deep internal Workbench module behind `canvas-engine/api.ts`, not another top-level Feature.                                                                                              |
| `image-actions`                                                                                      | Cross-surface image orchestration and transient display stay internal until a second independent implementation justifies a seam.                                                                     |

Overrides are longest-prefix rules, so a mixed folder's split is explicit rather than inferred from imports. Compatibility re-exports keep the target owner of the implementation they forward and are deletion work, never a fifth owner.

The owner dependency graph is acyclic: the only direction is `app -> workbench/features -> platform`, with cross-owner access through public module interfaces or composition-root adapters. `src/architecture/workbenchOwnership.test.ts` fails on any strongly connected component among owners, and `src/architecture/migrationExceptions.ts` is empty.

Feature interfaces remain ordinary module interfaces unless two adapters add leverage. Queue's backend port has production transport/realtime and test adapters; Identity's token/session port has browser-session and test adapters; Models and Nodes use production event/transport adapters; their sinks (model-load activity, node execution) are injected into Queue's runtime at the App composition root, with test doubles in Queue's runtime tests. Gallery, Generation, Workflow, and Upscale keep their compilation and UI seams internal until their feature tickets demonstrate another independent adapter. Workbench image actions likewise remain internal rather than becoming a shallow Image Feature. Gallery's picker and media slots read a narrow `GalleryHost` (gallery scope, notifications, an optional Gallery-widget reveal): the workbench derives it from the Gallery UI adapter, and the Launchpad's model manager, which has no workbench, supplies one itself. Media slots accept dragged gallery items only inside `GalleryDragScope`, which the workbench shell provides with its drag context; widget dialog hosts and the Launchpad sit outside it.

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
