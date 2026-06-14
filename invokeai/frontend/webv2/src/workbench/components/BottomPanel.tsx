import { useActiveProjectSelector } from '../WorkbenchContext';
import { getWidgetById } from '../widgetRegistry';
import { MissingWidgetFrame, WidgetRenderer } from './WidgetRenderer';

export const BottomPanel = () => {
  const panels = useActiveProjectSelector((project) => project.layout.panels);
  const bottomRegion = useActiveProjectSelector((project) => project.widgetRegions.bottom);
  const widget = getWidgetById(bottomRegion.activeWidgetId);
  const View = widget?.manifest.view;
  const canShowBottomPanel =
    panels.isBottomOpen &&
    !bottomRegion.isCollapsed &&
    bottomRegion.enabledWidgetIds.includes(bottomRegion.activeWidgetId) &&
    widget?.status === 'enabled' &&
    widget.manifest.bottomPanel !== 'tooltip';

  if (!canShowBottomPanel) {
    return null;
  }

  if (!widget || !View) {
    return <MissingWidgetFrame label={widget?.manifest.labelText ?? bottomRegion.activeWidgetId} region="bottom" />;
  }

  return <WidgetRenderer widget={widget} presentation="expanded" region="bottom" />;
};
