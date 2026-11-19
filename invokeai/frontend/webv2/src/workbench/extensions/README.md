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

Settings catalog is composed from first-party manifests, not live instance registration, so preferences stay discoverable while a widget is closed. Layered fields without a project or owning widget edit account defaults. An explicitly requested stale project/instance target is unavailable; it never creates a widget. Settings navigation carries the originating instance and returns focus to its gear on close.

## Discovery

The browser composes first-party widget manifests from its bundled registry. Server-discovered custom nodes are a separate system. There is no widget-pack installation, remote asset discovery, signing, permission, or reload protocol.

## Availability

Discovery (`getWidgetsForRegion`, `getWidgetHosts`; region menus, the Widgets dialog, the palette, placement commands) lists only available widgets. Two manifest rules exist, and they combine:

- `requiresAdmin`: a multi-user admin session. The authenticated route remounts on account changes, so this rule reads the session.
- `requiresCapability`: a server grant for the signed-in account (`WidgetCapability`), resolved by `workbench/widgetCapabilities.ts` from Identity's role and the account-scoped frontend config (`workbench/frontendConfig.ts`). Single-user mode counts as admin. A grant can change while the editor stays mounted. Renders that list a region holding a capability widget (today the status bar and the Widgets dialog) pass `useWidgetCapabilities()` to discovery; it also observes the config query, so whoever needs a grant loads it. Every other caller, commands included, reads the current grant from the query cache. The registry context function stays stable. A pending grant is unavailable.

Discovery is not the gate. A saved layout restores an instance by ID through `getWidgetById`, so a capability widget's own view checks the grant before doing any work. Without the grant it renders a localized refusal with its own way to leave the layout, since discovery also hides its chip. Example: [Server Output](../widgets/server-output/README.md).

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
