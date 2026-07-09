import type { LayerContextAction, LayerContextActionId } from './layerContextActions';

export type LayerContextSubmenuId = 'arrange' | 'add-modifiers' | 'add-regional' | 'boolean' | 'copy-to' | 'convert-to';

export type LayerContextMenuItem =
  | { action: LayerContextAction; kind: 'action' }
  | { actions: readonly LayerContextAction[]; id: LayerContextSubmenuId; kind: 'submenu' };

export interface LayerContextMenuSection {
  id: 'quick' | 'primary' | 'operations' | 'output' | 'state' | 'danger';
  items: readonly LayerContextMenuItem[];
  presentation: 'row' | 'list';
}

const ARRANGE_IDS: readonly LayerContextActionId[] = ['move-to-front', 'move-forward', 'move-backward', 'move-to-back'];
const MODIFIER_IDS: readonly LayerContextActionId[] = ['inpaint-noise', 'inpaint-denoise-limit'];
const REGIONAL_ADD_IDS: readonly LayerContextActionId[] = [
  'regional-positive-prompt',
  'regional-negative-prompt',
  'regional-reference-image',
];
const PRIMARY_IDS: readonly LayerContextActionId[] = [
  'transform',
  'rename',
  'fit-to-bbox',
  'filter',
  'control-transparency-effect',
  'regional-auto-negative',
  'extract-masked-area',
];
const BOOLEAN_IDS: readonly LayerContextActionId[] = ['intersect', 'cutout', 'cutaway', 'exclude'];
const COPY_IDS: readonly LayerContextActionId[] = [
  'copy-to-clipboard',
  'copy-to-raster',
  'copy-to-control',
  'copy-to-inpaint-mask',
  'copy-to-regional-guidance',
];
const CONVERT_IDS: readonly LayerContextActionId[] = [
  'rasterize',
  'convert-to-control',
  'convert-to-raster',
  'convert-to-inpaint-mask',
  'convert-to-regional-guidance',
];

const orderedActions = (
  byId: ReadonlyMap<LayerContextActionId, LayerContextAction>,
  ids: readonly LayerContextActionId[]
): LayerContextAction[] => ids.flatMap((id) => (byId.has(id) ? [byId.get(id)!] : []));

const actionItem = (
  byId: ReadonlyMap<LayerContextActionId, LayerContextAction>,
  id: LayerContextActionId
): LayerContextMenuItem[] => {
  const action = byId.get(id);
  return action ? [{ action, kind: 'action' }] : [];
};

const submenuItem = (
  byId: ReadonlyMap<LayerContextActionId, LayerContextAction>,
  id: LayerContextSubmenuId,
  actionIds: readonly LayerContextActionId[]
): LayerContextMenuItem[] => {
  const actions = orderedActions(byId, actionIds);
  return actions.length > 0 ? [{ actions, id, kind: 'submenu' }] : [];
};

/** Builds the legacy-style layer-menu hierarchy without coupling it to Chakra rendering. */
export const getLayerContextMenuLayout = (actions: readonly LayerContextAction[]): LayerContextMenuSection[] => {
  const byId = new Map(actions.map((action) => [action.id, action]));
  const sections: LayerContextMenuSection[] = [
    {
      id: 'quick',
      items: [...submenuItem(byId, 'arrange', ARRANGE_IDS), ...actionItem(byId, 'duplicate')],
      presentation: 'row',
    },
    {
      id: 'primary',
      items: [
        ...submenuItem(byId, 'add-modifiers', MODIFIER_IDS),
        ...submenuItem(byId, 'add-regional', REGIONAL_ADD_IDS),
        ...orderedActions(byId, PRIMARY_IDS).map((action) => ({ action, kind: 'action' }) as const),
      ],
      presentation: 'list',
    },
    {
      id: 'operations',
      items: [
        ...actionItem(byId, 'merge-down'),
        ...submenuItem(byId, 'boolean', BOOLEAN_IDS),
        ...submenuItem(byId, 'copy-to', COPY_IDS),
        ...submenuItem(byId, 'convert-to', CONVERT_IDS),
      ],
      presentation: 'list',
    },
    {
      id: 'output',
      items: [...actionItem(byId, 'crop-to-bbox'), ...actionItem(byId, 'save-to-assets')],
      presentation: 'list',
    },
    {
      id: 'state',
      items: [...actionItem(byId, 'toggle-visibility'), ...actionItem(byId, 'toggle-lock')],
      presentation: 'list',
    },
    { id: 'danger', items: actionItem(byId, 'delete'), presentation: 'list' },
  ];

  return sections.filter((section) => section.items.length > 0);
};
