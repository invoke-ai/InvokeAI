import { MissingWidgetFrame, WidgetRendererById } from '@workbench/widget-frame';
import { areWidgetRenderInstancesEqual } from '@workbench/widget-frame/widgetRenderInstance';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useActiveLayoutSelector, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useTranslation } from 'react-i18next';

export const BottomPanel = () => {
  const { t } = useTranslation();
  const bottomRegion = useActiveLayoutSelector((layout) => layout.widgetRegions.bottom);
  const instance = useActiveProjectSelector(
    (project) => project.widgetInstances[bottomRegion.activeInstanceId],
    areWidgetRenderInstancesEqual
  );
  const widget = instance ? getWidgetById(instance.typeId) : undefined;
  const canShowBottomPanel =
    !bottomRegion.isCollapsed &&
    bottomRegion.instanceIds.includes(bottomRegion.activeInstanceId) &&
    widget?.status === 'enabled' &&
    widget.manifest.bottomPanel !== 'tooltip' &&
    widget.manifest.bottomPanel !== 'popover';

  if (!canShowBottomPanel) {
    return null;
  }

  if (!instance || !widget) {
    return (
      <MissingWidgetFrame
        instanceId={bottomRegion.activeInstanceId}
        label={widget ? resolveWidgetLabel(widget.manifest, t) : bottomRegion.activeInstanceId}
        region="bottom"
        typeId={instance?.typeId}
      />
    );
  }

  return <WidgetRendererById instanceId={instance.id} widget={widget} presentation="expanded" region="bottom" />;
};
