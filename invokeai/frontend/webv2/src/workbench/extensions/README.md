# Workbench Widget Extensions

The workbench shell is now structured around widget types and widget instances.

- A widget manifest registers a widget type.
- A project layout stores widget instances in ordered region lists.
- A widget instance owns persisted project state.
- The shell owns placement, drag/drop, resizing, failure containment, and contribution APIs.

## Manifest Shape

Widgets should provide component icons, not icon IDs or import strings:

```tsx
import { LayersIcon } from 'lucide-react';

export const manifest = {
  id: 'invoke.layers',
  version: 1,
  label: 'Layers',
  labelText: 'Layers',
  icon: LayersIcon,
  allowedRegions: ['right'],
  allowMultiple: false,
  state: {
    version: 1,
    persistence: 'project',
    createInitial: () => ({}),
  },
  view: LayersWidgetView,
};
```

Third-party widgets should ship icons as JSX/SVG components in their compiled widget bundle.

## Discovery Options

Custom nodes today are server-discovered. The backend scans `custom_nodes_path`, skips hidden/underscore directories, requires `__init__.py`, and imports each pack with Python `importlib`. The management API can clone a git repo into that directory, reload nodes, and list installed packs.

Widgets should not mirror Python import execution directly in the browser. The safer server-first shape is:

- Add a configured `widgets_path` beside `custom_nodes_path`.
- Require each widget pack directory to include a manifest file, for example `invokeai-widget.json`.
- Let the server list widget packs and expose static compiled assets from each pack.
- The browser fetches a signed/validated widget catalog from the server.
- The shell dynamically imports approved widget entrypoints by URL only after the server has validated the pack manifest.

This avoids symlink assumptions and lets multi-user/admin rules follow the existing custom-node management pattern.

## Missing Core Systems

The extension API stubs exist in `extensionApi.ts`, but these systems still need real shell features:

- Command registry UI and lifecycle ownership.
- Command palette UI.
- Hotkey manager with context-aware `when` clauses.
- Global search UI and ranked result aggregation.
- Menu contribution rendering beyond widget header menus.
- Toolbar contribution rendering for locations like `center.tabs.trailing` and status bar slots.
- Extension install/uninstall/reload API for widget packs.
- Widget pack trust, signing, permission prompts, and admin policy.
- Extension asset serving and cache invalidation.

Undo/redo history controls were removed from normal widget registration because they render buttons and do not belong inside another button-like widget tab/slot. They should return as command and toolbar contributions when the command/toolbar systems exist.
