import { Box } from '@chakra-ui/react';
import { WorkflowImageExportProvider } from '@features/workflow/ui/nodeChrome';
import { Background, BackgroundVariant, ReactFlow, ReactFlowProvider, type ReactFlowProps } from '@xyflow/react';
import { useId, useMemo, type RefObject } from 'react';

import type { WorkflowFlowEdge, WorkflowFlowNode } from './flowAdapters';

/** A separate flow store keeps export measurements and expanded nodes out of the editor. */
export const WorkflowImageExportView = ({
  containerRef,
  ...props
}: ReactFlowProps<WorkflowFlowNode, WorkflowFlowEdge> & { containerRef: RefObject<HTMLDivElement | null> }) => {
  const gridId = useId();
  const nodes = useMemo(
    () => props.nodes?.map((node) => (node.selected ? { ...node, selected: false } : node)),
    [props.nodes]
  );
  // Every node draws open here, so every edge draws as itself: none hidden behind a collapsed edge, none merged.
  const edges = useMemo(
    () =>
      props.edges?.map(({ data, hidden: _hidden, ...edge }) => {
        const { collapsedEdgeIds: _collapsedEdgeIds, ...edgeData } = data ?? {};

        return {
          ...edge,
          animated: undefined,
          className: undefined,
          data: data ? (edgeData as typeof data) : data,
          selected: false,
          style: undefined,
        };
      }),
    [props.edges]
  );

  return (
    <Box ref={containerRef} aria-hidden inert h="full" left="-100000px" position="absolute" top="0" w="full">
      <ReactFlowProvider>
        <WorkflowImageExportProvider isExporting>
          <ReactFlow
            {...props}
            id={`workflow-export-${gridId}`}
            edges={edges}
            elementsSelectable={false}
            nodes={nodes}
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
