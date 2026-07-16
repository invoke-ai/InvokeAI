import type {
  LayerContextAction,
  LayerContextMenuSectionId,
  LayerContextSubmenuId,
  LayerType,
} from './layerContextActions';

export type { LayerContextSubmenuId } from './layerContextActions';

export type LayerContextMenuItem =
  | { action: LayerContextAction; kind: 'action' }
  | { actions: readonly LayerContextAction[]; id: LayerContextSubmenuId; kind: 'submenu' };

export interface LayerContextMenuSection {
  id: LayerContextMenuSectionId;
  items: readonly LayerContextMenuItem[];
  presentation: 'row' | 'list';
}

export type LayerContextMenuRenderEntry =
  | { id: 'before-danger'; kind: 'slot' }
  | { kind: 'section'; section: LayerContextMenuSection };

type LayerContextMenuLayerLabelKey =
  | 'widgets.canvas.import.control'
  | 'widgets.canvas.import.inpaintMask'
  | 'widgets.canvas.import.raster'
  | 'widgets.canvas.import.regionalGuidance';

const SECTION_LAYOUT: readonly {
  id: LayerContextMenuSectionId;
  order: number;
  presentation: 'row' | 'list';
}[] = [
  { id: 'quick', order: 0, presentation: 'row' },
  { id: 'primary', order: 1, presentation: 'list' },
  { id: 'operations', order: 2, presentation: 'list' },
  { id: 'output', order: 3, presentation: 'list' },
  { id: 'state', order: 4, presentation: 'list' },
  { id: 'danger', order: 5, presentation: 'list' },
];

const sortActions = (actions: readonly LayerContextAction[]): LayerContextAction[] =>
  [...actions].sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));

interface OrderedMenuItem {
  id: string;
  item: LayerContextMenuItem;
  order: number;
}

const buildSectionItems = (actions: readonly LayerContextAction[]): LayerContextMenuItem[] => {
  const ordered: OrderedMenuItem[] = [];
  const submenus = new Map<LayerContextSubmenuId, LayerContextAction[]>();

  for (const action of actions) {
    if (!action.submenu) {
      ordered.push({ id: action.id, item: { action, kind: 'action' }, order: action.order });
      continue;
    }
    const submenuActions = submenus.get(action.submenu) ?? [];
    submenuActions.push(action);
    submenus.set(action.submenu, submenuActions);
  }

  for (const [id, submenuActions] of submenus) {
    const sorted = sortActions(submenuActions);
    ordered.push({
      id,
      item: { actions: sorted, id, kind: 'submenu' },
      order: sorted[0]?.order ?? Number.POSITIVE_INFINITY,
    });
  }

  return ordered.sort((a, b) => a.order - b.order || a.id.localeCompare(b.id)).map(({ item }) => item);
};

/** Builds the layer-menu hierarchy exclusively from executable registry metadata. */
export const getLayerContextMenuLayout = (actions: readonly LayerContextAction[]): LayerContextMenuSection[] =>
  [...SECTION_LAYOUT]
    .sort((a, b) => a.order - b.order)
    .flatMap((section): LayerContextMenuSection[] => {
      const items = buildSectionItems(actions.filter((action) => action.section === section.id));
      return items.length > 0 ? [{ id: section.id, items, presentation: section.presentation }] : [];
    });

/** Inserts canvas-only items before destructive layer actions without coupling them to the layer-action registry. */
export const getLayerContextMenuRenderEntries = (
  sections: readonly LayerContextMenuSection[],
  includeBeforeDangerSlot: boolean
): LayerContextMenuRenderEntry[] =>
  sections.flatMap((section): LayerContextMenuRenderEntry[] =>
    includeBeforeDangerSlot && section.id === 'danger'
      ? [
          { id: 'before-danger', kind: 'slot' },
          { kind: 'section', section },
        ]
      : [{ kind: 'section', section }]
  );

/** Resolves the singular layer labels used by the legacy canvas context menu. */
export const getLayerContextMenuLayerLabelKey = (type: LayerType): LayerContextMenuLayerLabelKey => {
  switch (type) {
    case 'control':
      return 'widgets.canvas.import.control';
    case 'inpaint_mask':
      return 'widgets.canvas.import.inpaintMask';
    case 'raster':
      return 'widgets.canvas.import.raster';
    case 'regional_guidance':
      return 'widgets.canvas.import.regionalGuidance';
  }
};
