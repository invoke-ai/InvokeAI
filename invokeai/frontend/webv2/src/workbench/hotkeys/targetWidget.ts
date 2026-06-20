import type { WidgetInstanceId, WidgetTypeId } from '@workbench/types';

export const getHotkeyTargetWidget = (
  target: EventTarget | null
): { instanceId: WidgetInstanceId; typeId: WidgetTypeId } | null => {
  if (!(target instanceof Element)) {
    return null;
  }

  const widgetElement = target.closest('[data-hotkey-widget-instance-id][data-hotkey-widget-type-id]');
  const instanceId = widgetElement?.getAttribute('data-hotkey-widget-instance-id');
  const typeId = widgetElement?.getAttribute('data-hotkey-widget-type-id');

  return instanceId && typeId ? { instanceId, typeId: typeId as WidgetTypeId } : null;
};
