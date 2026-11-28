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

- `allowedRegions` says where the widget may be placed; any widget a side region may hold can also float in a window, with no opt-in.
- `needs` names the widget types this one acts on (Layers: `['canvas']`). Declare only what leaves the widget with nothing to act on, not a convenience. While a need is not on screen the shell shows a one-line notice in the widget's frame with the gesture that shows it, and navigation only enters workspaces that also place it ([needs](../widget-frame/README.md#needs)). The widget itself checks nothing.
- `graphBearing.sourceId` is the one mapping from a widget type to the invocation source it runs: `workbench/invocationSources.ts` derives both directions from the first-party manifests, read on first use. `graphBearing.surfaces` lists the regions where the widget renders a surface of its own (a workflow's graph in the center, its form in the left rail); a window shows the surface of the region it returns to, passed as `WidgetViewProps.surface`.

## Workbench API

`runtime.workbench.open(intent)` is for user gestures only: it may switch workspace. It shows a widget type where the current workspace places it, enters the workspace that does, or shows it here as a guest, and reports the outcome as an `OpenResult` ([how opening resolves](../widget-frame/README.md#opening-and-workspaces)); it fails only for a type that is unavailable. An intent names a type and optionally the surface it needs, never a region. `revealWidgetInstance(instanceId)` fronts an instance of the widget's own region. Widgets cannot add, move, float or remove widgets: that is structure, and only the shell's controls change it.

## Settings contributions

Optional `settings` contribution declares fields once and exposes a `quick` array of field IDs for the compact popover. All fields appear in the widget's full settings section and join search. Manifest with quick fields gets a shell-rendered popover plus “All settings” link; one without quick fields opens the full section directly. Widgets without preferences omit the contribution and the gear.

Field metadata: stable IDs, localized labels, optional descriptions/groups/search aliases, control type/options, storage scope. Deferred `load` returns a `Field` binding, which receives the field, presentation surface, and resolved project/instance target. Bindings use the owning module's reads and commands and shared Platform `SettingControl`. They must not create a widget view or engine just to edit a preference. Complex editors use the custom field kind and stay searchable as destinations.

Optional `WidgetImplementation.settingsActions` adds owner-rendered actions below quick preferences, e.g. Canvas diagnostics. These stay actions with their own availability checks, not persisted fields. Ordinary widget header actions stay available independent of settings.

Settings catalog is composed from first-party manifests, not live instance registration, so preferences stay discoverable while a widget is closed. Layered fields without a project or owning widget edit account defaults. An explicitly requested stale project/instance target is unavailable; it never creates a widget. Settings navigation carries the originating instance and returns focus to its gear on close.

## Discovery

The browser composes first-party widget manifests from its bundled registry. Server-discovered custom nodes are a separate system. There is no widget-pack installation, remote asset discovery, signing, permission, or reload protocol.

## Implemented Shell Systems

- Workbench and Launchpad command palette hosts with shared command and settings entries.
- First-party Workbench search providers for workflows, boards, models, images, queue items, prompt history.
- Widget command-palette contributions backed by the per-Workbench extension registry.
- Widget search-provider aggregation with source-safe identities and result adaptation into palette entries.
- Scoped provider search, fixed section ordering with ranked results inside each section, persistent command recents.
- Created-at date filtering for providers that explicitly advertise range support.

## Commands and hotkeys

Each Workbench mount owns its extension registry. Commands, hotkeys, palette entries and search providers carry their widget instance/project source and release their registrations on unmount. The hotkey runtime resolves current focus and editable/modal eligibility; Settings owns customized bindings and conflict presentation. Widget header actions and menus render from the loaded implementation in shell-owned slots.

There is no menu or toolbar contribution API. Shell layout controls and implementation header actions are the implemented surfaces; widget packs have no external compatibility contract.
