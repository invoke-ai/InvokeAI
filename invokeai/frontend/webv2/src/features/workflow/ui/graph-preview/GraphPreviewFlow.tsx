import type { XYPosition } from '@features/workflow/core/types';
import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';
import type { Edge as FlowEdge, Node as FlowNode, NodeProps, NodeTypes, ReactFlowInstance } from '@xyflow/react';
import type { TFunction } from 'i18next';

import { Badge, Box, Stack, Text } from '@chakra-ui/react';
import { getLayeredPositions } from '@features/workflow/core/graphLayout';
import '@xyflow/react/dist/style.css';
import { LOOP_LINKAGE_STROKE } from '@features/workflow/ui/loopLinkage';
import { useWorkflowPreferencesSelector } from '@features/workflow/ui/WorkflowUiContext';
import { Background, BackgroundVariant, Handle, Position, ReactFlow } from '@xyflow/react';
import { useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { flowThemeCss, getFlowColorMode } from './flowTheme';
import { getNodeSubtitle } from './nodeSummaries';

/**
 * The graph a library thumbnail photographs (`GraphPreviewSnapshot`): each node's type and a summary, fitted. Caller
 * position hints place nodes where given; the rest take a topological layout.
 */

type PreviewFlowNode = FlowNode<
  { inputCount: number; nodeId: string; nodeType: string; subtitle: string | null },
  'preview'
>;

const handleStyle = { background: 'var(--wb-flow-grid)', border: 'none' } as const;
const reactFlowProOptions = { hideAttribution: true } as const;
const reactFlowStyle = { background: 'transparent' } as const;

const PreviewNode = ({ data }: NodeProps<PreviewFlowNode>) => {
  const { t } = useTranslation();

  return (
    <Box bg="bg" borderColor="border.emphasized" borderWidth="1px" fontSize="md" minW="14rem" rounded="lg" shadow="sm">
      <Handle position={Position.Left} style={handleStyle} type="target" />
      <Handle position={Position.Right} style={handleStyle} type="source" />
      <Stack gap="0.5" px="3" py="2">
        <Badge fontFamily="mono" w="fit-content">
          {data.nodeType}
        </Badge>
        <Text color="fg.subtle" fontSize="xs" truncate>
          {data.subtitle ?? `${data.nodeId} · ${t('graphPreview.inputCount', { count: data.inputCount })}`}
        </Text>
      </Stack>
    </Box>
  );
};

const nodeTypes: NodeTypes = { preview: PreviewNode };

const toPreviewNodes = (
  graph: WorkflowPreviewGraph,
  t: TFunction,
  positionHints?: Record<string, XYPosition>
): PreviewFlowNode[] => {
  const positions = getLayeredPositions(
    graph.nodes,
    graph.edges.map((edge) => ({ sourceNodeId: edge.sourceNodeId, targetNodeId: edge.targetNodeId }))
  );

  return graph.nodes.map((node) => ({
    data: {
      inputCount: Object.keys(node.inputs).length,
      nodeId: node.id,
      nodeType: node.type,
      subtitle: getNodeSubtitle(node, t),
    },
    id: node.id,
    position: positionHints?.[node.id] ?? positions[node.id] ?? { x: 0, y: 0 },
    type: 'preview' as const,
  }));
};

const toPreviewEdges = (graph: WorkflowPreviewGraph): FlowEdge[] =>
  graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.sourceNodeId,
    target: edge.targetNodeId,
    type: 'default',
    ...(edge.type === 'loop_linkage'
      ? { style: { stroke: LOOP_LINKAGE_STROKE, strokeDasharray: '6 4', strokeWidth: 2 } }
      : {}),
  }));

export const GraphPreviewFlow = ({
  graph,
  positionHints,
  onInit,
}: {
  graph: WorkflowPreviewGraph;
  positionHints?: Record<string, XYPosition>;
  onInit: (instance: ReactFlowInstance) => void;
}) => {
  const { t } = useTranslation();
  const themeId = useWorkflowPreferencesSelector((preferences) => preferences.themeId);
  const backgroundId = useId().replace(/:/g, '');
  const nodes = useMemo(() => toPreviewNodes(graph, t, positionHints), [graph, t, positionHints]);
  const edges = useMemo(() => toPreviewEdges(graph), [graph]);
  // xyflow types `onInit` against this component's node/edge generics; the snapshot only calls generic instance
  // methods (`fitView`), so the narrower instance is safe to widen back to the public, type-erased signature.
  const handleInit = useCallback(
    (instance: ReactFlowInstance<PreviewFlowNode, FlowEdge>) => onInit(instance as unknown as ReactFlowInstance),
    [onInit]
  );

  return (
    <Box bg="bg.inset" css={flowThemeCss} h="full" rounded="md" w="full">
      <ReactFlow
        colorMode={getFlowColorMode(themeId)}
        edges={edges}
        edgesFocusable={false}
        elementsSelectable={false}
        fitView
        maxZoom={1.5}
        minZoom={0.1}
        nodes={nodes}
        nodesConnectable={false}
        nodesDraggable={false}
        nodesFocusable={false}
        nodeTypes={nodeTypes}
        proOptions={reactFlowProOptions}
        style={reactFlowStyle}
        onInit={handleInit}
      >
        <Background
          bgColor="var(--xy-background-color)"
          color="var(--wb-flow-grid)"
          gap={24}
          id={`preview-grid-${backgroundId}`}
          size={1.5}
          variant={BackgroundVariant.Dots}
        />
      </ReactFlow>
    </Box>
  );
};
