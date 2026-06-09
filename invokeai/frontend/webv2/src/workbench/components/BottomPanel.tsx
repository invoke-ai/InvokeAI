import { Text } from '@chakra-ui/react';

import { useWorkbench } from '../WorkbenchContext';
import { getWidgetById } from '../widgetRegistry';
import { WidgetPanelFrame } from './WidgetFrames';
import { WidgetRenderer } from './WidgetRenderer';

export const BottomPanel = () => {
  const { activeProject } = useWorkbench();
  const { panels } = activeProject.layout;
  const bottomRegion = activeProject.widgetRegions.bottom;
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
    return (
      <WidgetPanelFrame region="bottom">
        <Text color="fg.subtle" fontSize="2xs">
          Widget view unavailable.
        </Text>
      </WidgetPanelFrame>
    );
  }

  return (
    <WidgetPanelFrame region="bottom">
      <WidgetRenderer widget={widget} presentation="expanded" region="bottom" />
    </WidgetPanelFrame>
  );
};
