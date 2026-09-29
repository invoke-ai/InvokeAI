import { Box } from '@chakra-ui/react';
import { Background, BackgroundVariant, ReactFlow, ReactFlowProvider, type ReactFlowProps } from '@xyflow/react';
import { useId, type RefObject } from 'react';

import type { WorkflowFlowEdge, WorkflowFlowNode } from './flowAdapters';

import { WorkflowImageExportProvider } from './InvocationFlowNode';

/** A separate flow store keeps export measurements and expanded nodes out of the editor. */
export const WorkflowImageExportView = ({
  containerRef,
  ...props
}: ReactFlowProps<WorkflowFlowNode, WorkflowFlowEdge> & { containerRef: RefObject<HTMLDivElement | null> }) => {
  const gridId = useId();

  return (
    <Box ref={containerRef} aria-hidden inert h="full" left="-100000px" position="absolute" top="0" w="full">
      <ReactFlowProvider>
        <WorkflowImageExportProvider isExporting>
          <ReactFlow
            {...props}
            id={`workflow-export-${gridId}`}
            elementsSelectable={false}
            nodesConnectable={false}
            nodesDraggable={false}
          >
            <Background
              bgColor="var(--xy-background-color)"
              color="var(--wb-flow-grid)"
              gap={25}
              id={gridId}
              size={1.5}
              variant={BackgroundVariant.Dots}
            />
            {props.children}
          </ReactFlow>
        </WorkflowImageExportProvider>
      </ReactFlowProvider>
    </Box>
  );
};
