import type { WidgetRegion } from '@workbench/layoutContracts';
import type { NormalizedWidgetManifest, RegisteredWidget, WidgetInstanceId } from '@workbench/widgetContracts';
import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';
import type { LayoutAvailability } from '@workbench/workspace/resolveLayout';

import { canRenderWidgetInRegion, DEFAULT_ADD_REGIONS, isStatusChipWidget } from '@workbench/widgetPlacementCommands';
import { takesLastCenterView } from '@workbench/workspace/structureReducer';

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

/** Whether a placement open can show this widget; pure, so lists can ask on every render without chunk loads. */
export const canOpenWidget = (
  widget: RegisteredWidget,
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[]
): boolean =>
  widget.status === 'enabled' &&
  DEFAULT_ADD_REGIONS.some(
    (region) =>
      getWidgetsForRegion(region).some((candidate) => candidate.manifest.id === widget.manifest.id) &&
      canRenderWidgetInRegion(widget, region)
  );

/** Whether the list can put this widget back in the layout: as a panel or view, or as a status chip. Loads nothing. */
export const canAddWidget = (
  widget: RegisteredWidget,
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[]
): boolean => widget.status === 'enabled' && (isStatusChipWidget(widget) || canOpenWidget(widget, getWidgetsForRegion));

const findRegion = (project: WidgetPlacementProject, instanceId: WidgetInstanceId): WidgetRegion | undefined =>
  REGION_ORDER.find((region) => project.widgetRegions[region].instanceIds.includes(instanceId));

/** Placed instances first, then each widget type the layout lacks, so every widget can be brought back here. */
export const getWidgetsListRows = ({
  availability,
  getLabel,
  project,
  widgets,
}: {
  /** What marks the center's last view required (`takesLastCenterView`). */
  availability: Pick<LayoutAvailability, 'getPresentation'>;
  getLabel: (manifest: NormalizedWidgetManifest) => string;
  project: WidgetPlacementProject;
  /** Every widget this account can place, enabled ones only. */
  widgets: readonly RegisteredWidget[];
}): WidgetsListRow[] => {
  const byLabel = (left: WidgetsListRow, right: WidgetsListRow) =>
    left.label.localeCompare(right.label) ||
    getPlacementRank(left) - getPlacementRank(right) ||
    (left.instanceId ?? '').localeCompare(right.instanceId ?? '');
  const placed = Object.values(project.widgetInstances).flatMap((instance): WidgetsListRow[] => {
    const widget = widgets.find((candidate) => candidate.manifest.id === instance.typeId);
    const region = project.floatingPlacements?.[instance.id] ? undefined : findRegion(project, instance.id);

    if (!widget || (!region && !project.floatingPlacements?.[instance.id])) {
      return [];
    }

    return [
      {
        instanceId: instance.id,
        isRequired: region === 'center' && takesLastCenterView(project, instance.id, availability),
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
