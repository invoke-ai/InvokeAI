import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo } from 'react';

import { useAutoLayoutContext } from './auto-layout-context';

export const PanelHotkeysLogical = memo(() => {
  const { tab } = useAutoLayoutContext();

  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleLeftPanel',
    callback: () => {
      if (navigationApi.tabApi?.getTab() !== tab) {
        return;
      }
      navigationApi.toggleLeftPanelInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleRightPanel',
    callback: () => {
      if (navigationApi.tabApi?.getTab() !== tab) {
        return;
      }
      navigationApi.toggleRightPanelInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'resetPanelLayout',
    callback: () => {
      if (navigationApi.tabApi?.getTab() !== tab) {
        return;
      }
      navigationApi.resetPanelsInTab(tab);
    },
    dependencies: [tab],
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'togglePanels',
    callback: () => {
      if (navigationApi.tabApi?.getTab() !== tab) {
        return;
      }
      navigationApi.toggleBothPanelsInTab(tab);
    },
    dependencies: [tab],
  });

  return null;
});

PanelHotkeysLogical.displayName = 'PanelHotkeysLogical';
