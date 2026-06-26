import type { WidgetInstanceId, WidgetTypeId, WorkbenchRegion } from '@workbench/types';

export const getHotkeyTargetWidget = (
  target: EventTarget | null
): { instanceId: WidgetInstanceId; region: WorkbenchRegion | null; typeId: WidgetTypeId } | null => {
  if (!(target instanceof Element)) {
    return null;
  }

  const widgetElement = target.closest('[data-hotkey-widget-instance-id][data-hotkey-widget-type-id]');
  const instanceId = widgetElement?.getAttribute('data-hotkey-widget-instance-id');
  const region = widgetElement?.getAttribute('data-hotkey-widget-region') ?? null;
  const typeId = widgetElement?.getAttribute('data-hotkey-widget-type-id');

  return instanceId && typeId
    ? { instanceId, region: region as WorkbenchRegion | null, typeId: typeId as WidgetTypeId }
    : null;
};
