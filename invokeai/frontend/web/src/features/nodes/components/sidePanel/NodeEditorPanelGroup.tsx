import 'reactflow/dist/style.css';

import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useCallback, useRef } from 'react';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import InspectorPanel from './inspector/InspectorPanel';
import { WorkflowViewMode } from './viewMode/WorkflowViewMode';
import WorkflowPanel from './workflow/WorkflowPanel';
import { WorkflowListMenu } from './WorkflowListMenu/WorkflowListMenu';
import { WorkflowListMenuTrigger } from './WorkflowListMenu/WorkflowListMenuTrigger';

const panelGroupStyles: CSSProperties = { height: '100%', width: '100%' };

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const NodeEditorPanelGroup = () => {
  const mode = useAppSelector(selectWorkflowMode);
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const workflowListMenu = useWorkflowListMenu();

  const handleDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }
    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      <WorkflowListMenuTrigger />
      <Flex w="full" h="full" position="relative">
        <Box position="absolute" top={0} left={0} right={0} bottom={0}>
          {workflowListMenu.isOpen && (
            <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
              <Flex gap={2} flexDirection="column" h="full" w="full">
                <WorkflowListMenu />
              </Flex>
            </OverlayScrollbarsComponent>
          )}

          <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
            {mode === 'view' && <WorkflowViewMode />}
            {mode === 'edit' && (
              <PanelGroup
                ref={panelGroupRef}
                id="workflow-panel-group"
                autoSaveId="workflow-panel-group"
                direction="vertical"
                style={panelGroupStyles}
              >
                <Panel id="workflow" collapsible minSize={25}>
                  <WorkflowPanel />
                </Panel>
                <ResizeHandle onDoubleClick={handleDoubleClickHandle} />
                <Panel id="inspector" collapsible minSize={25}>
                  <InspectorPanel />
                </Panel>
              </PanelGroup>
            )}
          </OverlayScrollbarsComponent>
        </Box>
      </Flex>
    </Flex>
  );
};

export default memo(NodeEditorPanelGroup);
