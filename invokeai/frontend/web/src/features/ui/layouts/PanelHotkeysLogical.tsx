import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';

import { useAutoLayoutContext } from './auto-layout-context';
import { panelRegistry } from './panel-registry/panelApiRegistry';

export const PanelHotkeysLogical = memo(() => {
  const { tab } = useAutoLayoutContext();

  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleLeftPanel',
    callback: () => {
      if (panelRegistry.tabApi?.getTab() !== tab) {
        return;
      }
      panelRegistry.toggleLeftPanelInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleRightPanel',
    callback: () => {
      if (panelRegistry.tabApi?.getTab() !== tab) {
        return;
      }
      panelRegistry.toggleRightPanelInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'resetPanelLayout',
    callback: () => {
      if (panelRegistry.tabApi?.getTab() !== tab) {
        return;
      }
      panelRegistry.resetPanelsInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'togglePanels',
    callback: () => {
      if (panelRegistry.tabApi?.getTab() !== tab) {
        return;
      }
      panelRegistry.toggleBothPanelsInTab(tab);
    },
    dependencies: [tab],
  });

  return null;
});

PanelHotkeysLogical.displayName = 'PanelHotkeysLogical';
