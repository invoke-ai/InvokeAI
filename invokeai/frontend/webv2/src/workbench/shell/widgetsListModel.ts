import type { WidgetRegion } from '@workbench/layoutContracts';
import type { NormalizedWidgetManifest, RegisteredWidget, WidgetInstanceId } from '@workbench/widgetContracts';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';
import type { WorkbenchWidgetCommands } from '@workbench/workbenchStore';

import {
  getEnabledCenterViewCount,
  getOpenableWidgetPlacement,
  openWidgetPlacement,
} from '@workbench/widgetPlacementCommands';

/** Where a widget is, as the Widgets list says it: a docked region, a floating window, or nowhere in this layout. */
export type WidgetsListPlacement = { kind: 'docked'; region: WidgetRegion } | { kind: 'floating' } | { kind: 'hidden' };

export interface WidgetsListRow {
  /** The instance for a placed row; null for a widget type this layout does not hold. */
  instanceId: WidgetInstanceId | null;
  label: string;
  placement: WidgetsListPlacement;
  /** Why Remove is refused: the center always shows one view. */
  isRequired: boolean;
  widget: RegisteredWidget;
}

/** A docked instance is named by the first region in this order that holds it: where the user most likely sees it. */
const REGION_ORDER: readonly WidgetRegion[] = ['center', 'left', 'right', 'bottom'];

/** Rows that read the same keep one order: by where they are, then by instance. */
const PLACEMENT_RANK: Record<string, number> = { bottom: 3, center: 0, floating: 4, hidden: 5, left: 1, right: 2 };

const getPlacementRank = ({ placement }: WidgetsListRow): number =>
  PLACEMENT_RANK[placement.kind === 'docked' ? placement.region : placement.kind]!;

/**
 * A status chip: a widget that lives only as a chip in the status bar (a tooltip or popover there), which placement
 * opens refuse because it renders no panel.
 */
const isStatusChip = (widget: RegisteredWidget): boolean =>
  widget.manifest.allowedRegions.includes('bottom') &&
  (widget.manifest.bottomPanel === 'tooltip' || widget.manifest.bottomPanel === 'popover');

/** Whether the list can put this widget back in the layout: as a panel or view, or as a status chip. */
export const canAddWidget = (
  widget: RegisteredWidget,
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[]
): boolean =>
  widget.status === 'enabled' &&
  (getOpenableWidgetPlacement({ getWidgetsForRegion, typeId: widget.manifest.id }) !== null || isStatusChip(widget));

/** Puts a widget in the layout where it opens by default; a status chip goes back to the status bar. */
export const addWidgetToLayout = ({
  getWidgetsForRegion,
  widget,
  widgets,
}: {
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  widget: RegisteredWidget;
  widgets: WorkbenchWidgetCommands;
}): boolean => {
  const { allowMultiple, id: typeId, state } = widget.manifest;

  if (
    openWidgetPlacement({
      getWidgetsForRegion,
      options: allowMultiple ? { createNew: true } : undefined,
      typeId,
      widgets,
    }).ok
  ) {
    return true;
  }

  if (widget.status !== 'enabled' || !isStatusChip(widget)) {
    return false;
  }

  widgets.open({
    createNew: allowMultiple ? true : undefined,
    initialValues: state.createInitial(),
    region: 'bottom',
    widgetId: typeId,
  });

  return true;
};

const findRegion = (project: WidgetPlacementProject, instanceId: WidgetInstanceId): WidgetRegion | undefined =>
  REGION_ORDER.find((region) => project.widgetRegions[region].instanceIds.includes(instanceId));

/**
 * One row per placed instance, then one per widget type the layout holds none of (or that allows more), so every
 * widget can be found and brought back from one list. Placed rows come first, each group by label.
 */
export const getWidgetsListRows = ({
  getLabel,
  getWidgetById,
  project,
  widgets,
}: {
  getLabel: (manifest: NormalizedWidgetManifest) => string;
  getWidgetById: (typeId: string) => RegisteredWidget | undefined;
  project: WidgetPlacementProject;
  /** Every widget this account can place, enabled ones only. */
  widgets: readonly RegisteredWidget[];
}): WidgetsListRow[] => {
  const byLabel = (left: WidgetsListRow, right: WidgetsListRow) =>
    left.label.localeCompare(right.label) ||
    getPlacementRank(left) - getPlacementRank(right) ||
    (left.instanceId ?? '').localeCompare(right.instanceId ?? '');
  const centerViewCount = getEnabledCenterViewCount(project, getWidgetById);
  const placed = Object.values(project.widgetInstances).flatMap((instance): WidgetsListRow[] => {
    const widget = widgets.find((candidate) => candidate.manifest.id === instance.typeId);
    const region = project.floatingPlacements?.[instance.id] ? undefined : findRegion(project, instance.id);

    if (!widget || (!region && !project.floatingPlacements?.[instance.id])) {
      return [];
    }

    return [
      {
        instanceId: instance.id,
        isRequired: region === 'center' && widget.manifest.centerPlacement !== 'toolbar' && centerViewCount === 1,
        label: instance.title ?? getLabel(widget.manifest),
        placement: region ? { kind: 'docked', region } : { kind: 'floating' },
        widget,
      },
    ];
  });
  const placedTypeIds = new Set(placed.map((row) => row.widget.manifest.id));
  const hidden = widgets
    .filter((widget) => widget.manifest.allowMultiple || !placedTypeIds.has(widget.manifest.id))
    .map((widget): WidgetsListRow => ({
      instanceId: null,
      isRequired: false,
      label: getLabel(widget.manifest),
      placement: { kind: 'hidden' },
      widget,
    }));

  return [...placed.sort(byLabel), ...hidden.sort(byLabel)];
};
