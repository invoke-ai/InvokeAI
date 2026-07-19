import type { WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';
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
}

export interface AvailableWidgetTypeItem extends BaseWidgetRegionItem {
  instance?: undefined;
  isEnabled: false;
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

export const createWidgetRegionViewModel = <Instance extends WidgetPlacementInstanceMeta>({
  activeInstanceId,
  instanceIds,
  region,
  widgetInstances,
  widgets,
  getWidgetLabel = (manifest) => (typeof manifest.label === 'string' ? manifest.label : manifest.id),
}: {
  activeInstanceId?: WidgetInstanceId;
  instanceIds: WidgetInstanceId[];
  region: WidgetRegion;
  widgetInstances: Record<string, Instance>;
  widgets: RegisteredWidget[];
  getWidgetLabel?: (manifest: NormalizedWidgetManifest) => string;
}): WidgetRegionViewModel<Instance> => {
  const widgetsByType = new Map(widgets.map((widget) => [widget.manifest.id, widget]));
  const placedItems = instanceIds.flatMap((instanceId): PlacedWidgetRegionItem<Instance>[] => {
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
        label: instance.title ?? getWidgetLabel(widget.manifest),
        status: widget.status,
        typeId: instance.typeId,
        widget,
      },
    ];
  });
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
    sortableInstanceIds: placedItems.map((item) => item.id),
  };
};

export const createWidgetRegionViewModelFromState = <Instance extends WidgetPlacementInstanceMeta>({
  region,
  regionState,
  widgetInstances,
  widgets,
  getWidgetLabel,
}: {
  region: WidgetRegion;
  regionState: WidgetRegionState;
  widgetInstances: Record<string, Instance>;
  widgets: RegisteredWidget[];
  getWidgetLabel?: (manifest: NormalizedWidgetManifest) => string;
}): WidgetRegionViewModel<Instance> =>
  createWidgetRegionViewModel({
    activeInstanceId: regionState.activeInstanceId,
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
): boolean => item.widget.manifest.bottomPanel !== 'tooltip';

export const canRemoveItem = (item: WidgetRegionItem, viewModel: WidgetRegionViewModel): boolean => {
  if (viewModel.region !== 'center') {
    return item.isEnabled;
  }

  const enabledCenterViewCount = viewModel.placedItems.filter(
    (placedItem) => placedItem.status === 'enabled' && placedItem.widget.manifest.centerPlacement !== 'toolbar'
  ).length;

  return !isRequiredCenterView(item, enabledCenterViewCount);
};
