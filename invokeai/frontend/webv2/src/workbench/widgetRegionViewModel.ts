import type { FloatingWidgetState, WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
import type {
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetIconComponent,
  WidgetInstanceContract,
  WidgetInstanceId,
  WidgetTypeId,
} from '@workbench/widgetContracts';

export interface WidgetPlacementInstanceMeta {
  id: WidgetInstanceId;
  typeId: WidgetTypeId;
  title?: string;
}

export type WidgetPlacementMeta = Record<WidgetInstanceId, WidgetPlacementInstanceMeta>;

interface BaseWidgetRegionItem {
  failureMessage?: string;
  allowMultiple: boolean;
  icon: WidgetIconComponent;
  id: string;
  label: string;
  status: RegisteredWidget['status'];
  typeId: WidgetTypeId;
  widget: RegisteredWidget;
}

export interface PlacedWidgetRegionItem<
  Instance extends WidgetPlacementInstanceMeta = WidgetInstanceContract,
> extends BaseWidgetRegionItem {
  instance: Instance;
  isEnabled: true;
  /** Detached into a floating window; the rail keeps its slot, at the position it docks back to. */
  isFloating?: true;
}

/** Only the placement half of a window: the slot this region keeps for it. */
export type FloatingWidgetPlacement = Pick<FloatingWidgetState, 'returnIndex' | 'returnRegion'>;

export interface AvailableWidgetTypeItem extends BaseWidgetRegionItem {
  instance?: undefined;
  isEnabled: false;
  isFloating?: undefined;
}

export type WidgetRegionItem<Instance extends WidgetPlacementInstanceMeta = WidgetInstanceContract> =
  | PlacedWidgetRegionItem<Instance>
  | AvailableWidgetTypeItem;

export interface WidgetRegionViewModel<Instance extends WidgetPlacementInstanceMeta = WidgetInstanceContract> {
  region: WidgetRegion;
  placedItems: PlacedWidgetRegionItem<Instance>[];
  availableItems: AvailableWidgetTypeItem[];
  activeItem?: PlacedWidgetRegionItem<Instance>;
  sortableInstanceIds: WidgetInstanceId[];
}

/**
 * Keep floated widgets reachable from their original rail slots; insert by ascending dock index to preserve
 * ordering.
 */
const withFloatingSlots = (
  instanceIds: WidgetInstanceId[],
  region: WidgetRegion,
  floatingWidgets: Record<WidgetInstanceId, FloatingWidgetPlacement> | undefined
): { instanceId: WidgetInstanceId; isFloating: boolean }[] => {
  const slots = instanceIds.map((instanceId) => ({ instanceId, isFloating: false }));

  if (!floatingWidgets) {
    return slots;
  }

  const floating = (Object.entries(floatingWidgets) as [WidgetInstanceId, FloatingWidgetPlacement][])
    .filter(([instanceId, state]) => state.returnRegion === region && !instanceIds.includes(instanceId))
    .map(([instanceId, state]) => ({ index: state.returnIndex ?? Number.POSITIVE_INFINITY, instanceId }))
    .sort((left, right) => left.index - right.index);

  for (const { index, instanceId } of floating) {
    slots.splice(Math.min(Math.max(0, Math.floor(index)), slots.length), 0, { instanceId, isFloating: true });
  }

  return slots;
};

export const createWidgetRegionViewModel = <Instance extends WidgetPlacementInstanceMeta>({
  activeInstanceId,
  floatingWidgets,
  instanceIds,
  region,
  widgetInstances,
  widgets,
  getWidgetLabel = (manifest) => (typeof manifest.label === 'string' ? manifest.label : manifest.id),
}: {
  activeInstanceId?: WidgetInstanceId;
  /** Windows floated out of regions; those returning here keep a rail slot. */
  floatingWidgets?: Record<WidgetInstanceId, FloatingWidgetPlacement>;
  instanceIds: WidgetInstanceId[];
  region: WidgetRegion;
  widgetInstances: Record<string, Instance>;
  widgets: RegisteredWidget[];
  getWidgetLabel?: (manifest: NormalizedWidgetManifest) => string;
}): WidgetRegionViewModel<Instance> => {
  const widgetsByType = new Map(widgets.map((widget) => [widget.manifest.id, widget]));
  const placedItems = withFloatingSlots(instanceIds, region, floatingWidgets).flatMap(
    ({ instanceId, isFloating }): PlacedWidgetRegionItem<Instance>[] => {
      const instance = widgetInstances[instanceId];
      const widget = instance ? widgetsByType.get(instance.typeId) : undefined;

      if (!instance || !widget) {
        return [];
      }

      return [
        {
          failureMessage: widget.failure?.message,
          allowMultiple: widget.manifest.allowMultiple,
          icon: widget.manifest.icon,
          id: instance.id,
          instance,
          isEnabled: true,
          ...(isFloating ? { isFloating: true } : {}),
          label: instance.title ?? getWidgetLabel(widget.manifest),
          status: widget.status,
          typeId: instance.typeId,
          widget,
        },
      ];
    }
  );
  const placedTypeIds = new Set(placedItems.map((item) => item.typeId));
  const availableItems: AvailableWidgetTypeItem[] = widgets
    .filter((widget) => widget.manifest.allowMultiple || !placedTypeIds.has(widget.manifest.id))
    .map((widget) => ({
      failureMessage: widget.failure?.message,
      allowMultiple: widget.manifest.allowMultiple,
      icon: widget.manifest.icon,
      id: `${region}:new:${widget.manifest.id}`,
      isEnabled: false,
      label: getWidgetLabel(widget.manifest),
      status: widget.status,
      typeId: widget.manifest.id,
      widget,
    }));
  const activeItem = placedItems.find((item) => item.id === activeInstanceId);

  return {
    activeItem,
    availableItems,
    placedItems,
    region,
    // A floating slot is a pointer to the window, not a tab: it neither drags nor
    // gives the strip a drop index, so it stays out of the sortable list.
    sortableInstanceIds: placedItems.filter((item) => !item.isFloating).map((item) => item.id),
  };
};

export const createWidgetRegionViewModelFromState = <Instance extends WidgetPlacementInstanceMeta>({
  floatingWidgets,
  region,
  regionState,
  widgetInstances,
  widgets,
  getWidgetLabel,
}: {
  floatingWidgets?: Record<WidgetInstanceId, FloatingWidgetPlacement>;
  region: WidgetRegion;
  regionState: WidgetRegionState;
  widgetInstances: Record<string, Instance>;
  widgets: RegisteredWidget[];
  getWidgetLabel?: (manifest: NormalizedWidgetManifest) => string;
}): WidgetRegionViewModel<Instance> =>
  createWidgetRegionViewModel({
    activeInstanceId: regionState.activeInstanceId,
    floatingWidgets,
    getWidgetLabel,
    instanceIds: regionState.instanceIds,
    region,
    widgetInstances,
    widgets,
  });

export const getWidgetRegionItems = <Instance extends WidgetPlacementInstanceMeta>(
  viewModel: WidgetRegionViewModel<Instance>
): WidgetRegionItem<Instance>[] => [...viewModel.placedItems, ...viewModel.availableItems];

export const isPlacedWidgetRegionItem = <Instance extends WidgetPlacementInstanceMeta>(
  item: WidgetRegionItem<Instance>
): item is PlacedWidgetRegionItem<Instance> => item.isEnabled;

export const isRequiredCenterView = (item: WidgetRegionItem, enabledCenterViewCount: number): boolean => {
  const isView = item.widget.manifest.centerPlacement !== 'toolbar';

  return isView && item.isEnabled && enabledCenterViewCount === 1;
};

export const isCompactBottomItem = <Instance extends WidgetPlacementInstanceMeta>(
  item: WidgetRegionItem<Instance>
): item is PlacedWidgetRegionItem<Instance> => item.isEnabled && item.status === 'enabled';

export const isExpandableBottomItem = <Instance extends WidgetPlacementInstanceMeta>(
  item: WidgetRegionItem<Instance>
): boolean => item.widget.manifest.bottomPanel !== 'tooltip' && item.widget.manifest.bottomPanel !== 'popover';

export const isPopoverBottomItem = <Instance extends WidgetPlacementInstanceMeta>(
  item: WidgetRegionItem<Instance>
): boolean => item.widget.manifest.bottomPanel === 'popover';

export const canRemoveItem = (item: WidgetRegionItem, viewModel: WidgetRegionViewModel): boolean => {
  if (viewModel.region !== 'center') {
    return item.isEnabled;
  }

  const enabledCenterViewCount = viewModel.placedItems.filter(
    (placedItem) => placedItem.status === 'enabled' && placedItem.widget.manifest.centerPlacement !== 'toolbar'
  ).length;

  return !isRequiredCenterView(item, enabledCenterViewCount);
};
