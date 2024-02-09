import 'reactflow/dist/style.css';

import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import QueueControls from 'features/queue/components/QueueControls';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { usePanelStorage } from 'features/ui/hooks/usePanelStorage';
import type { CSSProperties } from 'react';
import { memo, useCallback, useRef } from 'react';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import InspectorPanel from './inspector/InspectorPanel';
import { ModeToggle } from './ModeToggle';
import { WorkflowViewMode } from './viewMode/WorkflowViewMode';
import WorkflowPanel from './workflow/WorkflowPanel';
import { WorkflowMenu } from './WorkflowMenu';
import { WorkflowName } from './WorkflowName';

const panelGroupStyles: CSSProperties = { height: '100%', width: '100%' };

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
  return {
    mode: workflow.mode,
  };
});

const NodeEditorPanelGroup = () => {
  const { mode } = useAppSelector(selector);
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const panelStorage = usePanelStorage();

  const handleDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }
    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      <QueueControls />
      <Flex w="full" justifyContent="space-between" alignItems="center" gap="4" padding={1}>
        <WorkflowName />
        <WorkflowMenu />
      </Flex>
      <ModeToggle />

      {mode === 'view' && <WorkflowViewMode />}
      {mode === 'edit' && (
        <PanelGroup
          ref={panelGroupRef}
          id="workflow-panel-group"
          autoSaveId="workflow-panel-group"
          direction="vertical"
          style={panelGroupStyles}
          storage={panelStorage}
        >
          <Panel id="workflow" collapsible minSize={25}>
            <WorkflowPanel />
          </Panel>
          <ResizeHandle orientation="horizontal" onDoubleClick={handleDoubleClickHandle} />
          <Panel id="inspector" collapsible minSize={25}>
            <InspectorPanel />
          </Panel>
        </PanelGroup>
      )}
    </Flex>
  );
};

export default memo(NodeEditorPanelGroup);
