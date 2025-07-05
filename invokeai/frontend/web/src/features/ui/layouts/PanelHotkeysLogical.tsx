import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo } from 'react';

export const PanelHotkeysLogical = memo(() => {
  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleLeftPanel',
    callback: navigationApi.toggleLeftPanel,
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'toggleRightPanel',
    callback: navigationApi.toggleRightPanel,
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'resetPanelLayout',
    callback: navigationApi.resetLeftAndRightPanels,
  });
  useRegisteredHotkeys({
    category: 'app',
    id: 'togglePanels',
    callback: navigationApi.toggleLeftAndRightPanels,
  });

  return null;
});

PanelHotkeysLogical.displayName = 'PanelHotkeysLogical';
