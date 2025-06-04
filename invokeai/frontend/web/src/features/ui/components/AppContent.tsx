import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasMainPanelContent } from 'features/controlLayers/components/CanvasMainPanelContent';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import WorkflowsTabLeftPanel from 'features/nodes/components/sidePanel/WorkflowsTabLeftPanel';
import QueueControls from 'features/queue/components/QueueControls';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { FloatingLeftPanelButtons } from 'features/ui/components/FloatingLeftPanelButtons';
import { FloatingRightPanelButtons } from 'features/ui/components/FloatingRightPanelButtons';
import ParametersPanelTextToImage from 'features/ui/components/ParametersPanels/ParametersPanelTextToImage';
import { RightPanelContent } from 'features/ui/components/RightPanelContent';
import ModelManagerTab from 'features/ui/components/tabs/ModelManagerTab';
import QueueTab from 'features/ui/components/tabs/QueueTab';
import { WorkflowsMainPanel } from 'features/ui/components/tabs/WorkflowsTabContent';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import type { UsePanelOptions } from 'features/ui/hooks/usePanel';
import { usePanel } from 'features/ui/hooks/usePanel';
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
import { Panel, PanelGroup } from 'react-resizable-panels';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

import ParametersPanelUpscale from './ParametersPanels/ParametersPanelUpscale';
import { VerticalResizeHandle } from './tabs/ResizeHandle';

const panelStyles: CSSProperties = { position: 'relative', height: '100%', width: '100%', minWidth: 0 };

const onLeftPanelCollapse = (isCollapsed: boolean) => $isLeftPanelOpen.set(!isCollapsed);
const onRightPanelCollapse = (isCollapsed: boolean) => $isRightPanelOpen.set(!isCollapsed);

export const AppContent = memo(() => {
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
    <Flex id="invoke-app-tabs" w="full" h="full" gap={4} p={4} overflow="hidden">
      <VerticalNavBar />
      <PanelGroup
        ref={imperativePanelGroupRef}
        id="app-panel-group"
        autoSaveId="app-panel-group"
        direction="horizontal"
        style={panelStyles}
      >
        {withLeftPanel && (
          <>
            <Panel id="left-panel" order={0} collapsible style={panelStyles} {...leftPanel.panelProps}>
              <LeftPanelContent />
            </Panel>
            <VerticalResizeHandle id="left-main-handle" {...leftPanel.resizeHandleProps} />
          </>
        )}
        <Panel id="main-panel" order={1} minSize={20} style={panelStyles}>
          <MainPanelContent />
          {withLeftPanel && <FloatingLeftPanelButtons onToggle={leftPanel.toggle} />}
          {withRightPanel && <FloatingRightPanelButtons onToggle={rightPanel.toggle} />}
        </Panel>
        {withRightPanel && (
          <>
            <VerticalResizeHandle id="main-right-handle" {...rightPanel.resizeHandleProps} />
            <Panel id="right-panel" order={2} style={panelStyles} collapsible {...rightPanel.panelProps}>
              <RightPanelContent />
            </Panel>
          </>
        )}
      </PanelGroup>
    </Flex>
  );
});
AppContent.displayName = 'AppContent';

const LeftPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  return (
    <Flex flexDir="column" w="full" h="full" gap={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        {tab === 'canvas' && <ParametersPanelTextToImage />}
        {tab === 'upscaling' && <ParametersPanelUpscale />}
        {tab === 'workflows' && <WorkflowsTabLeftPanel />}
      </Box>
    </Flex>
  );
});
LeftPanelContent.displayName = 'LeftPanelContent';

const MainPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  if (tab === 'canvas') {
    return <CanvasMainPanelContent />;
  }
  if (tab === 'upscaling') {
    return <ImageViewer />;
  }
  if (tab === 'workflows') {
    return <WorkflowsMainPanel />;
  }
  if (tab === 'models') {
    return <ModelManagerTab />;
  }
  if (tab === 'queue') {
    return <QueueTab />;
  }

  assert<Equals<never, typeof tab>>(false);
});
MainPanelContent.displayName = 'MainPanelContent';
