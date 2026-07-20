import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';

import { Badge, Box, Stack, Text } from '@chakra-ui/react';
import { isInvocationNode, type ProjectGraphState, type XYPosition } from '@features/workflow/contracts';
import '@xyflow/react/dist/style.css';
import { useWorkflowPreferencesSelector } from '@features/workflow/ui/WorkflowUiContext';
import { getResolvedWorkflowEdges } from '@features/workflow/utility';
import {
  Background,
  BackgroundVariant,
  Handle,
  Position,
  ReactFlow,
  type Edge as FlowEdge,
  type Node as FlowNode,
  type NodeProps,
  type NodeTypes,
} from '@xyflow/react';
import { useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { flowThemeCss, getFlowColorMode } from './flowTheme';

/**
 * A read-only flow rendering of a compiled `GraphContract` — the "nodes" half
 * of the graph preview dialog. Contracts carry no positions, so the renderer
 * uses the caller's hints (the editable document's positions, when the graph
 * came from the project graph) and falls back to a layered topological layout.
 */

type PreviewFlowNode = FlowNode<{ inputCount: number; nodeId: string; nodeType: string }, 'preview'>;

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

const LAYER_WIDTH = 300;
const ROW_HEIGHT = 100;
const handleStyle = { background: 'var(--wb-flow-grid)', border: 'none' } as const;
const reactFlowProOptions = { hideAttribution: true } as const;
const reactFlowStyle = { background: 'transparent' } as const;

const PreviewNode = ({ data }: NodeProps<PreviewFlowNode>) => {
  const { t } = useTranslation();

  return (
    <Box bg="bg" borderColor="border.emphasized" borderWidth="1px" fontSize="xs" minW="14rem" rounded="lg" shadow="sm">
      <Handle position={Position.Left} style={handleStyle} type="target" />
      <Handle position={Position.Right} style={handleStyle} type="source" />
      <Stack gap="0.5" px="3" py="2">
        <Badge fontFamily="mono" size="xs" w="fit-content">
          {data.nodeType}
        </Badge>
        <Text color="fg.subtle" fontSize="2xs" truncate>
          {data.nodeId} · {t('graphPreview.inputCount', { count: data.inputCount })}
        </Text>
      </Stack>
    </Box>
  );
};

const nodeTypes: NodeTypes = { preview: PreviewNode };

/** Longest-path-from-roots depth per node; cycle members settle at their first depth. */
const getNodeDepths = (graph: WorkflowPreviewGraph): Map<string, number> => {
  const depths = new Map<string, number>();
  const incoming = new Map<string, string[]>();

  for (const edge of graph.edges) {
    incoming.set(edge.targetNodeId, [...(incoming.get(edge.targetNodeId) ?? []), edge.sourceNodeId]);
  }

  const resolve = (nodeId: string, seen: Set<string>): number => {
    const known = depths.get(nodeId);

    if (known !== undefined) {
      return known;
    }

    if (seen.has(nodeId)) {
      return 0;
    }

    seen.add(nodeId);

    const parents = incoming.get(nodeId) ?? [];
    const depth = parents.length === 0 ? 0 : Math.max(...parents.map((parent) => resolve(parent, seen))) + 1;

    depths.set(nodeId, depth);

    return depth;
  };

  for (const node of graph.nodes) {
    resolve(node.id, new Set());
  }

  return depths;
};

const toPreviewNodes = (graph: WorkflowPreviewGraph, positionHints?: Record<string, XYPosition>): PreviewFlowNode[] => {
  const depths = getNodeDepths(graph);
  const rowsPerDepth = new Map<number, number>();

  return graph.nodes.map((node) => {
    const hint = positionHints?.[node.id];

    if (hint) {
      return {
        data: { inputCount: Object.keys(node.inputs).length, nodeId: node.id, nodeType: node.type },
        id: node.id,
        position: hint,
        type: 'preview' as const,
      };
    }

    const depth = depths.get(node.id) ?? 0;
    const row = rowsPerDepth.get(depth) ?? 0;

    rowsPerDepth.set(depth, row + 1);

    return {
      data: { inputCount: Object.keys(node.inputs).length, nodeId: node.id, nodeType: node.type },
      id: node.id,
      position: { x: depth * LAYER_WIDTH, y: row * ROW_HEIGHT },
      type: 'preview' as const,
    };
  });
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
}: {
  graph: WorkflowPreviewGraph;
  positionHints?: Record<string, XYPosition>;
}) => {
  const themeId = useWorkflowPreferencesSelector((preferences) => preferences.themeId);
  const backgroundId = useId().replace(/:/g, '');
  const nodes = useMemo(() => toPreviewNodes(graph, positionHints), [graph, positionHints]);
  const edges = useMemo(() => toPreviewEdges(graph), [graph]);

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
