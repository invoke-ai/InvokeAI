import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';
import type {
  Edge as FlowEdge,
  Node as FlowNode,
  NodeMouseHandler,
  NodeProps,
  NodeTypes,
  ReactFlowInstance,
} from '@xyflow/react';
import type { TFunction } from 'i18next';

import { Badge, Box, Stack, Text } from '@chakra-ui/react';
import { isInvocationNode, type ProjectGraphState, type XYPosition } from '@features/workflow/contracts';
import { getLayeredPositions } from '@features/workflow/core/graphLayout';
import '@xyflow/react/dist/style.css';
import { useWorkflowPreferencesSelector } from '@features/workflow/ui/WorkflowUiContext';
import { getResolvedWorkflowEdges } from '@features/workflow/utility';
import { Background, BackgroundVariant, Handle, Position, ReactFlow } from '@xyflow/react';
import { useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { flowThemeCss, getFlowColorMode } from './flowTheme';
import { getNodeSubtitle } from './nodeSummaries';

/**
 * A read-only flow rendering of a compiled `GraphContract` — the "nodes" half
 * of the graph preview dialog. Contracts carry no positions, so the renderer
 * uses the caller's hints (the editable document's positions, when the graph
 * came from the project graph) and falls back to a layered topological layout.
 */

type PreviewFlowNode = FlowNode<
  { inputCount: number; isSelected: boolean; nodeId: string; nodeType: string; subtitle: string | null },
  'preview'
>;

/**
 * A lightweight contract view of an editable document (no templates needed),
 * for previewing workflows that are not the active project graph — e.g.
 * library entries before loading them.
 */
export const documentToPreviewGraph = (
  document: ProjectGraphState,
  fallbackLabel: string
): { graph: WorkflowPreviewGraph; positionHints: Record<string, XYPosition> } => {
  const invocationNodes = document.nodes.filter(isInvocationNode);
  const invocationNodeIds = new Set(invocationNodes.map((node) => node.id));

  return {
    graph: {
      edges: getResolvedWorkflowEdges(document.nodes, document.edges)
        .filter((edge) => invocationNodeIds.has(edge.source) && invocationNodeIds.has(edge.target))
        .map((edge) => ({
          id: edge.id,
          sourceField: edge.sourceHandle,
          sourceNodeId: edge.source,
          targetField: edge.targetHandle,
          targetNodeId: edge.target,
        })),
      id: document.id,
      label: document.name || fallbackLabel,
      nodes: invocationNodes.map((node) => ({
        id: node.id,
        inputs: Object.fromEntries(Object.values(node.data.inputs).map((instance) => [instance.name, instance.value])),
        type: node.data.type,
      })),
      updatedAt: document.updatedAt,
      version: 1,
    },
    positionHints: Object.fromEntries(document.nodes.map((node) => [node.id, node.position])),
  };
};

const handleStyle = { background: 'var(--wb-flow-grid)', border: 'none' } as const;
const reactFlowProOptions = { hideAttribution: true } as const;
const reactFlowStyle = { background: 'transparent' } as const;

const PreviewNode = ({ data }: NodeProps<PreviewFlowNode>) => {
  const { t } = useTranslation();

  return (
    <Box
      bg="bg"
      borderColor={data.isSelected ? 'accent.emphasized' : 'border.emphasized'}
      borderWidth="1px"
      fontSize="xs"
      minW="14rem"
      rounded="lg"
      shadow="sm"
    >
      <Handle position={Position.Left} style={handleStyle} type="target" />
      <Handle position={Position.Right} style={handleStyle} type="source" />
      <Stack gap="0.5" px="3" py="2">
        <Badge fontFamily="mono" size="xs" w="fit-content">
          {data.nodeType}
        </Badge>
        <Text color="fg.subtle" fontSize="2xs" truncate>
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
  positionHints?: Record<string, XYPosition>,
  selectedNodeId?: string | null
): PreviewFlowNode[] => {
  const positions = getLayeredPositions(
    graph.nodes,
    graph.edges.map((edge) => ({ sourceNodeId: edge.sourceNodeId, targetNodeId: edge.targetNodeId }))
  );

  return graph.nodes.map((node) => ({
    data: {
      inputCount: Object.keys(node.inputs).length,
      isSelected: node.id === selectedNodeId,
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
  }));

export const GraphPreviewFlow = ({
  graph,
  positionHints,
  selectedNodeId = null,
  onInit,
  onNodeSelect,
}: {
  graph: WorkflowPreviewGraph;
  positionHints?: Record<string, XYPosition>;
  selectedNodeId?: string | null;
  onInit?: (instance: ReactFlowInstance) => void;
  onNodeSelect?: (nodeId: string | null) => void;
}) => {
  const { t } = useTranslation();
  const themeId = useWorkflowPreferencesSelector((preferences) => preferences.themeId);
  const backgroundId = useId().replace(/:/g, '');
  const nodes = useMemo(
    () => toPreviewNodes(graph, t, positionHints, selectedNodeId),
    [graph, t, positionHints, selectedNodeId]
  );
  const edges = useMemo(() => toPreviewEdges(graph), [graph]);
  const handleNodeClick = useCallback<NodeMouseHandler<PreviewFlowNode>>(
    (_event, node) => onNodeSelect?.(node.id),
    [onNodeSelect]
  );
  const handlePaneClick = useCallback(() => onNodeSelect?.(null), [onNodeSelect]);
  // xyflow types `onInit` against this component's node/edge generics; the caller
  // (`GraphPreviewDialog`) only calls generic instance methods (`fitView`), so the
  // narrower instance is safe to widen back to the public, type-erased signature.
  const handleInit = useCallback(
    (instance: ReactFlowInstance<PreviewFlowNode, FlowEdge>) => onInit?.(instance as unknown as ReactFlowInstance),
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
        onNodeClick={handleNodeClick}
        onPaneClick={handlePaneClick}
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
