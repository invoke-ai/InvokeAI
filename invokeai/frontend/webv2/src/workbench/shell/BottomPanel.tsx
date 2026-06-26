import { MissingWidgetFrame, WidgetRendererById } from '@workbench/widget-frame';
import { areWidgetRenderInstancesEqual } from '@workbench/widget-frame/widgetRenderInstance';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';

export const BottomPanel = () => {
  const panels = useActiveProjectSelector((project) => project.layout.panels);
  const bottomRegion = useActiveProjectSelector((project) => project.widgetRegions.bottom);
  const instance = useActiveProjectSelector(
    (project) => project.widgetInstances[bottomRegion.activeInstanceId],
    areWidgetRenderInstancesEqual
  );
  const widget = instance ? getWidgetById(instance.typeId) : undefined;
  const View = widget?.manifest.view;
  const canShowBottomPanel =
    panels.isBottomOpen &&
    !bottomRegion.isCollapsed &&
    bottomRegion.instanceIds.includes(bottomRegion.activeInstanceId) &&
    widget?.status === 'enabled' &&
    widget.manifest.bottomPanel !== 'tooltip';

  if (!canShowBottomPanel) {
    return null;
  }

  if (!instance || !widget || !View) {
    return <MissingWidgetFrame label={widget?.manifest.labelText ?? bottomRegion.activeInstanceId} region="bottom" />;
  }

  return <WidgetRendererById instanceId={instance.id} widget={widget} presentation="expanded" region="bottom" />;
};
