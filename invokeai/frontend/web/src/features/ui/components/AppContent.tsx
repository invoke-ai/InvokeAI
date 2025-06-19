import 'dockview/dist/styles/dockview.css';
import 'features/ui/styles/dockview-theme-invoke.css';

import { Flex } from '@invoke-ai/ui-library';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import { AutoLayout } from 'features/ui/layouts/AutoLayout';
import { $isLeftPanelOpen, $isRightPanelOpen } from 'features/ui/store/uiSlice';
import type { CSSProperties } from 'react';
import { memo } from 'react';

const panelStyles: CSSProperties = { position: 'relative', height: '100%', width: '100%', minWidth: 0 };

const onLeftPanelCollapse = (isCollapsed: boolean) => $isLeftPanelOpen.set(!isCollapsed);
const onRightPanelCollapse = (isCollapsed: boolean) => $isRightPanelOpen.set(!isCollapsed);

export const AppContent = memo(() => {
  // const tab = useAppSelector(selectActiveTab);
  // const imperativePanelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  useDndMonitor();

  // const withLeftPanel = useAppSelector(selectWithLeftPanel);
  // const leftPanelUsePanelOptions = useMemo<UsePanelOptions>(
  //   () => ({
  //     id: 'left-panel',
  //     minSizePx: LEFT_PANEL_MIN_SIZE_PX,
  //     defaultSizePx: LEFT_PANEL_MIN_SIZE_PX,
  //     imperativePanelGroupRef,
  //     panelGroupDirection: 'horizontal',
  //     onCollapse: onLeftPanelCollapse,
  //   }),
  //   []
  // );
  // const leftPanel = usePanel(leftPanelUsePanelOptions);
  // useRegisteredHotkeys({
  //   id: 'toggleLeftPanel',
  //   category: 'app',
  //   callback: leftPanel.toggle,
  //   options: { enabled: withLeftPanel },
  //   dependencies: [leftPanel.toggle, withLeftPanel],
  // });

  // const withRightPanel = useAppSelector(selectWithRightPanel);
  // const rightPanelUsePanelOptions = useMemo<UsePanelOptions>(
  //   () => ({
  //     id: 'right-panel',
  //     minSizePx: RIGHT_PANEL_MIN_SIZE_PX,
  //     defaultSizePx: RIGHT_PANEL_MIN_SIZE_PX,
  //     imperativePanelGroupRef,
  //     panelGroupDirection: 'horizontal',
  //     onCollapse: onRightPanelCollapse,
  //   }),
  //   []
  // );
  // const rightPanel = usePanel(rightPanelUsePanelOptions);
  // useRegisteredHotkeys({
  //   id: 'toggleRightPanel',
  //   category: 'app',
  //   callback: rightPanel.toggle,
  //   options: { enabled: withRightPanel },
  //   dependencies: [rightPanel.toggle, withRightPanel],
  // });

  // useRegisteredHotkeys({
  //   id: 'resetPanelLayout',
  //   category: 'app',
  //   callback: () => {
  //     leftPanel.reset();
  //     rightPanel.reset();
  //   },
  //   dependencies: [leftPanel.reset, rightPanel.reset],
  // });
  // useRegisteredHotkeys({
  //   id: 'togglePanels',
  //   category: 'app',
  //   callback: () => {
  //     if (leftPanel.isCollapsed || rightPanel.isCollapsed) {
  //       leftPanel.expand();
  //       rightPanel.expand();
  //     } else {
  //       leftPanel.collapse();
  //       rightPanel.collapse();
  //     }
  //   },
  //   dependencies: [
  //     leftPanel.isCollapsed,
  //     rightPanel.isCollapsed,
  //     leftPanel.expand,
  //     rightPanel.expand,
  //     leftPanel.collapse,
  //     rightPanel.collapse,
  //   ],
  // });

  return (
    <Flex id="invoke-app-tabs" w="full" h="full" p={0} overflow="hidden">
      <VerticalNavBar />
      <AutoLayout />
    </Flex>
  );
});
AppContent.displayName = 'AppContent';
