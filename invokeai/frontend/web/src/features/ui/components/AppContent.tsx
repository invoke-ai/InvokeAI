import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
import { CanvasMainPanelContent } from 'features/controlLayers/components/CanvasMainPanelContent';
import { CanvasRightPanel } from 'features/controlLayers/components/CanvasRightPanel';
import GalleryPanelContent from 'features/gallery/components/GalleryPanelContent';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import NodeEditorPanelGroup from 'features/nodes/components/sidePanel/NodeEditorPanelGroup';
import QueueControls from 'features/queue/components/QueueControls';
import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import ParametersPanelTextToImage from 'features/ui/components/ParametersPanels/ParametersPanelTextToImage';
import ModelManagerTab from 'features/ui/components/tabs/ModelManagerTab';
import QueueTab from 'features/ui/components/tabs/QueueTab';
import { WorkflowsTabContent } from 'features/ui/components/tabs/WorkflowsTabContent';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import type { UsePanelOptions } from 'features/ui/hooks/usePanel';
import { usePanel } from 'features/ui/hooks/usePanel';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import {
  $isLeftPanelOpen,
  $isRightPanelOpen,
  LEFT_PANEL_MIN_SIZE_PCT,
  LEFT_PANEL_MIN_SIZE_PX,
  RIGHT_PANEL_MIN_SIZE_PCT,
  RIGHT_PANEL_MIN_SIZE_PX,
  selectWithLeftPanel,
  selectWithRightPanel,
} from 'features/ui/store/uiSlice';
import type { CSSProperties } from 'react';
import { memo, useMemo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import ParametersPanelUpscale from './ParametersPanels/ParametersPanelUpscale';
import ResizeHandle from './tabs/ResizeHandle';

const panelStyles: CSSProperties = { position: 'relative', height: '100%', width: '100%' };

const onLeftPanelCollapse = (isCollapsed: boolean) => $isLeftPanelOpen.set(!isCollapsed);
const onRightPanelCollapse = (isCollapsed: boolean) => $isRightPanelOpen.set(!isCollapsed);

export const AppContent = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  useScopeOnFocus('gallery', ref);

  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);

  const withLeftPanel = useAppSelector(selectWithLeftPanel);
  const leftPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'left-panel',
      unit: 'pixels',
      minSize: LEFT_PANEL_MIN_SIZE_PX,
      defaultSize: LEFT_PANEL_MIN_SIZE_PCT,
      panelGroupRef,
      panelGroupDirection: 'horizontal',
      onCollapse: onLeftPanelCollapse,
    }),
    []
  );
  const leftPanel = usePanel(leftPanelUsePanelOptions);
  useHotkeys(['t', 'o'], leftPanel.toggle, { enabled: withLeftPanel }, [leftPanel.toggle, withLeftPanel]);

  const withRightPanel = useAppSelector(selectWithRightPanel);
  const rightPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      id: 'right-panel',
      unit: 'pixels',
      minSize: RIGHT_PANEL_MIN_SIZE_PX,
      defaultSize: RIGHT_PANEL_MIN_SIZE_PCT,
      panelGroupRef,
      panelGroupDirection: 'horizontal',
      onCollapse: onRightPanelCollapse,
    }),
    []
  );
  const rightPanel = usePanel(rightPanelUsePanelOptions);
  useHotkeys('g', rightPanel.toggle, { enabled: withRightPanel }, [rightPanel.toggle, withRightPanel]);

  useHotkeys(
    'shift+r',
    () => {
      leftPanel.reset();
      rightPanel.reset();
    },
    [leftPanel.reset, rightPanel.reset]
  );
  useHotkeys(
    'f',
    () => {
      if (leftPanel.isCollapsed || rightPanel.isCollapsed) {
        leftPanel.expand();
        rightPanel.expand();
      } else {
        leftPanel.collapse();
        rightPanel.collapse();
      }
    },
    [
      leftPanel.isCollapsed,
      rightPanel.isCollapsed,
      leftPanel.expand,
      rightPanel.expand,
      leftPanel.collapse,
      rightPanel.collapse,
    ]
  );

  return (
    <Flex ref={ref} id="invoke-app-tabs" w="full" h="full" gap={4} p={4}>
      <VerticalNavBar />
      <Flex position="relative" w="full" h="full" gap={4}>
        <PanelGroup
          ref={panelGroupRef}
          id="app-panel-group"
          autoSaveId="app-panel-group"
          direction="horizontal"
          style={panelStyles}
        >
          {withLeftPanel && (
            <>
              <Panel order={0} collapsible style={panelStyles} {...leftPanel.panelProps}>
                <Flex flexDir="column" w="full" h="full" gap={2}>
                  <QueueControls />
                  <Box position="relative" w="full" h="full">
                    <LeftPanelContent />
                  </Box>
                </Flex>
              </Panel>
              <ResizeHandle id="left-main-handle" orientation="vertical" {...leftPanel.resizeHandleProps} />
            </>
          )}
          <Panel id="main-panel" order={1} minSize={20} style={panelStyles}>
            <MainPanelContent />
          </Panel>
          {withRightPanel && (
            <>
              <ResizeHandle id="main-right-handle" orientation="vertical" {...rightPanel.resizeHandleProps} />
              <Panel order={2} style={panelStyles} collapsible {...rightPanel.panelProps}>
                <RightPanelContent />
              </Panel>
            </>
          )}
        </PanelGroup>
        {withLeftPanel && <FloatingParametersPanelButtons panelApi={leftPanel} />}
        {withRightPanel && <FloatingGalleryButton panelApi={rightPanel} />}
      </Flex>
    </Flex>
  );
});

AppContent.displayName = 'AppContent';

const RightPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  if (tab === 'generation') {
    return <CanvasRightPanel />;
  }

  if (tab === 'upscaling' || tab === 'workflows') {
    return <GalleryPanelContent />;
  }

  return null;
});
RightPanelContent.displayName = 'RightPanelContent';

const LeftPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);

  if (tab === 'generation') {
    return <ParametersPanelTextToImage />;
  }
  if (tab === 'upscaling') {
    return <ParametersPanelUpscale />;
  }

  if (tab === 'workflows') {
    return <NodeEditorPanelGroup />;
  }

  return null;
});
LeftPanelContent.displayName = 'LeftPanelContent';

const MainPanelContent = memo(() => {
  const tab = useAppSelector(selectActiveTab);
  if (tab === 'generation') {
    return <CanvasMainPanelContent />;
  }
  if (tab === 'upscaling') {
    return <ImageViewer />;
  }
  if (tab === 'workflows') {
    return <WorkflowsTabContent />;
  }
  if (tab === 'models') {
    return <ModelManagerTab />;
  }
  if (tab === 'queue') {
    return <QueueTab />;
  }

  return null;
});
MainPanelContent.displayName = 'MainPanelContent';
