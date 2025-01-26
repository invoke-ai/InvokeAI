import { Box } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { ViewContextProvider } from 'features/nodes/contexts/ViewContext';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import type { CSSProperties } from 'react';
import { memo, useCallback, useRef } from 'react';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import WorkflowNodeInspectorPanel from './inspector/WorkflowNodeInspectorPanel';
import WorkflowFieldsLinearViewPanel from './workflow/WorkflowPanel';

const panelGroupStyles: CSSProperties = { height: '100%', width: '100%' };

export const EditModeLeftPanelContent = memo(() => {
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);

  const handleDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }
    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  return (
    <ViewContextProvider view="edit-mode-linear">
      <Box position="relative" w="full" h="full">
        <ScrollableContent>
          <PanelGroup
            ref={panelGroupRef}
            id="workflow-panel-group"
            autoSaveId="workflow-panel-group"
            direction="vertical"
            style={panelGroupStyles}
          >
            <Panel id="workflow" collapsible minSize={25}>
              <WorkflowFieldsLinearViewPanel />
            </Panel>
            <ResizeHandle onDoubleClick={handleDoubleClickHandle} />
            <Panel id="inspector" collapsible minSize={25}>
              <WorkflowNodeInspectorPanel />
            </Panel>
          </PanelGroup>
        </ScrollableContent>
      </Box>
    </ViewContextProvider>
  );
});

EditModeLeftPanelContent.displayName = 'EditModeLeftPanelContent';
