# Workbench Widget Extensions

Workbench shell now built around widget types and widget instances.

- Widget manifest registers a widget type.
- Project layout stores widget instances in ordered region lists.
- Widget instance owns persisted project state.
- Shell owns placement, drag/drop, resizing, failure containment, contribution APIs.

## Manifest Shape

Widgets should provide component icons, not icon IDs or import strings:

```tsx
import type { WidgetManifest } from '@workbench/widgetContracts';
import { LayersIcon } from 'lucide-react';
import { layerSettingsContribution } from './settingsContribution';

export const manifest: WidgetManifest = {
  id: 'invoke.layers',
  version: 1,
  label: 'Layers',
  icon: LayersIcon,
  allowedRegions: ['right'],
  allowMultiple: false,
  state: {
    version: 1,
    persistence: 'project',
    createInitial: () => ({}),
  },
  load: () => import('./implementation'),
  settings: layerSettingsContribution,
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
};
```

Third-party widgets should ship icons as JSX/SVG components in their compiled widget bundle.

## Settings contributions

Optional `settings` contribution declares fields once and exposes a `quick` array of field IDs for the compact popover. All fields appear in the widget's full settings section and join search. Manifest with quick fields gets a shell-rendered popover plus “All settings” link; one without quick fields opens the full section directly. Widgets without preferences omit the contribution and the gear.

Field metadata: stable IDs, localized labels, optional descriptions/groups/search aliases, control type/options, storage scope. Deferred `load` returns a `Field` binding, which receives the field, presentation surface, and resolved project/instance target. Bindings use the owning module's reads and commands and shared Platform `SettingControl`. They must not create a widget view or engine just to edit a preference. Complex editors use the custom field kind and stay searchable as destinations.

Optional `WidgetImplementation.settingsActions` adds owner-rendered actions below quick preferences, e.g. Canvas diagnostics. These stay actions with their own availability checks, not persisted fields. Ordinary widget header actions stay available independent of settings.

Settings catalog is composed from first-party manifests, not live instance registration, so preferences stay discoverable while a widget is closed. Missing project or instance is an explicit unavailable state, not a request to create one. Settings navigation carries the originating instance and returns focus to its gear on close.

## Discovery Options

Custom nodes today are server-discovered. Backend scans `custom_nodes_path`, skips hidden/underscore directories, requires `__init__.py`, imports each pack with Python `importlib`. Management API can clone a git repo into that directory, reload nodes, list installed packs.

Widgets should not mirror Python import execution directly in the browser. Safer server-first shape:

- Add configured `widgets_path` beside `custom_nodes_path`.
- Require each widget pack directory to include a manifest file, e.g. `invokeai-widget.json`.
- Server lists widget packs and exposes static compiled assets from each pack.
- Browser fetches a signed/validated widget catalog from the server.
- Shell dynamically imports approved widget entrypoints by URL only after the server validated the pack manifest.

This avoids symlink assumptions and lets multi-user/admin rules follow the existing custom-node management pattern.

## Implemented Shell Systems

- Workbench and Launchpad command palette hosts with shared command and settings entries.
- First-party Workbench search providers for workflows, boards, models, images, queue items, prompt history.
- Widget command-palette contributions backed by the per-Workbench extension registry.
- Widget search-provider aggregation with source-safe identities and result adaptation into palette entries.
- Scoped provider search, fixed section ordering with ranked results inside each section, persistent command recents.
- Created-at date filtering for providers that explicitly advertise range support.

## Missing Core Systems

Contribution registries exist in `extensionRegistry.ts` (one registry per Workbench mount, constructed by `WorkbenchProvider`), but these systems still need real shell features:

- Command registry UI and lifecycle ownership.
- Hotkey manager with context-aware `when` clauses.
- Menu contribution rendering. `menus.register` stores contributions, but nothing reads them yet, widget header menus included.
- Toolbar contribution rendering for locations like `center.tabs.trailing` and status bar slots.
- Extension install/uninstall/reload API for widget packs.
- Widget pack trust, signing, permission prompts, admin policy.
- Extension asset serving and cache invalidation.

Undo/redo history controls were removed from normal widget registration: they render buttons and do not belong inside another button-like widget tab/slot. They should return as command and toolbar contributions once command/toolbar systems exist.
