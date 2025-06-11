import 'dockview/dist/styles/dockview.css';
import 'features/ui/styles/dockview-theme-invoke.css';

import { TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import type { UsePanelOptions } from 'features/ui/hooks/usePanel';
import { usePanel } from 'features/ui/hooks/usePanel';
import { CanvasTabAutoLayout } from 'features/ui/layouts/canvas-tab-auto-layout';
import { GenerateTabAutoLayout } from 'features/ui/layouts/generate-tab-auto-layout';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import {
  $isLeftPanelOpen,
  $isRightPanelOpen,
  LEFT_PANEL_MIN_SIZE_PX,
  RIGHT_PANEL_MIN_SIZE_PX,
  selectWithLeftPanel,
  selectWithRightPanel,
} from 'features/ui/store/uiSlice';
import type { CSSProperties } from 'react';
import { memo, useMemo, useRef } from 'react';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';

const panelStyles: CSSProperties = { position: 'relative', height: '100%', width: '100%', minWidth: 0 };

const onLeftPanelCollapse = (isCollapsed: boolean) => $isLeftPanelOpen.set(!isCollapsed);
const onRightPanelCollapse = (isCollapsed: boolean) => $isRightPanelOpen.set(!isCollapsed);

export const AppContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);
  const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  useDndMonitor();

  const withLeftPanel = useAppSelector(selectWithLeftPanel);
  const leftPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'left-panel',
      minSizePx: LEFT_PANEL_MIN_SIZE_PX,
      defaultSizePx: LEFT_PANEL_MIN_SIZE_PX,
      imperativePanelGroupRef,
      panelGroupDirection: 'horizontal',
      onCollapse: onLeftPanelCollapse,
    }),
    []
  );
  const leftPanel = usePanel(leftPanelUsePanelOptions);
  useRegisteredHotkeys({
    id: 'toggleLeftPanel',
    category: 'app',
    callback: leftPanel.toggle,
    options: { enabled: withLeftPanel },
    dependencies: [leftPanel.toggle, withLeftPanel],
  });

  const withRightPanel = useAppSelector(selectWithRightPanel);
  const rightPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'right-panel',
      minSizePx: RIGHT_PANEL_MIN_SIZE_PX,
      defaultSizePx: RIGHT_PANEL_MIN_SIZE_PX,
      imperativePanelGroupRef,
      panelGroupDirection: 'horizontal',
      onCollapse: onRightPanelCollapse,
    }),
    []
  );
  const rightPanel = usePanel(rightPanelUsePanelOptions);
  useRegisteredHotkeys({
    id: 'toggleRightPanel',
    category: 'app',
    callback: rightPanel.toggle,
    options: { enabled: withRightPanel },
    dependencies: [rightPanel.toggle, withRightPanel],
  });

  useRegisteredHotkeys({
    id: 'resetPanelLayout',
    category: 'app',
    callback: () => {
      leftPanel.reset();
      rightPanel.reset();
    },
    dependencies: [leftPanel.reset, rightPanel.reset],
  });
  useRegisteredHotkeys({
    id: 'togglePanels',
    category: 'app',
    callback: () => {
      if (leftPanel.isCollapsed || rightPanel.isCollapsed) {
        leftPanel.expand();
        rightPanel.expand();
      } else {
        leftPanel.collapse();
        rightPanel.collapse();
      }
    },
    dependencies: [
      leftPanel.isCollapsed,
      rightPanel.isCollapsed,
      leftPanel.expand,
      rightPanel.expand,
      leftPanel.collapse,
      rightPanel.collapse,
    ],
  });

  return (
    <Tabs index={tab === 'generate' ? 0 : 1} variant="unstyled" w="full" h="full" display="flex" p={0}>
      <TabList>
        <VerticalNavBar />
      </TabList>
      <TabPanels w="full" h="full" p={0}>
        <TabPanel w="full" h="full" p={0}>
          <GenerateTabAutoLayout />
        </TabPanel>
        <TabPanel w="full" h="full" p={0}>
          <CanvasTabAutoLayout />
        </TabPanel>
      </TabPanels>
    </Tabs>
  );
});
AppContent.displayName = 'AppContent';
