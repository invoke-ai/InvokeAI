import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { omit, reduce } from 'es-toolkit/compat';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import type { BoardField } from 'features/nodes/types/common';
import type { BoardFieldInputInstance } from 'features/nodes/types/field';
import { isBoardFieldInputInstance, isBoardFieldInputTemplate } from 'features/nodes/types/field';
import type { InvocationNodeData } from 'features/nodes/types/invocation';
import { isBatchNodeType, isGeneratorNodeType } from 'features/nodes/types/invocation';
import type { AnyInvocation, Graph } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

const log = logger('workflows');

type BoardFieldResolver = (field: BoardFieldInputInstance) => BoardField | undefined;

const getBoardField = (field: BoardFieldInputInstance, state: RootState): BoardField | undefined => {
  // Translate the UI value to the graph value. See note in BoardFieldInputComponent for more info.
  const { value } = field;

  if (value === 'auto' || !value) {
    const autoAddBoardId = selectAutoAddBoardId(state);
    if (autoAddBoardId === 'none') {
      return undefined;
    }
    return {
      board_id: autoAddBoardId,
    };
  }

  if (value === 'none') {
    return undefined;
  }

  return value;
};

const defaultBoardFieldResolver: BoardFieldResolver = (field) => {
  const { value } = field;
  if (!value || value === 'none' || value === 'auto') {
    return undefined;
  }
  return value;
};

type NodeLike = {
  id: string;
  type?: string;
  data?: unknown;
};

type InvocationNodeLike = NodeLike & {
  data: InvocationNodeData;
};

const isInvocationNodeLike = (node: NodeLike): node is InvocationNodeLike => {
  if (node.type !== 'invocation') {
    return false;
  }

  if (!node.data || typeof node.data !== 'object') {
    return false;
  }

  const data = node.data as Partial<InvocationNodeData>;
  return Boolean(data.inputs && data.type && data.useCache !== undefined && data.isIntermediate !== undefined);
};

type EdgeLike = {
  type?: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};

type BuildInvocationGraphArgs = {
  nodes: NodeLike[];
  edges: EdgeLike[];
  templates: Templates;
  resolveBoardField?: BoardFieldResolver;
  graphId?: string;
};

export const buildInvocationGraph = ({
  nodes,
  edges,
  templates,
  resolveBoardField = defaultBoardFieldResolver,
  graphId,
}: BuildInvocationGraphArgs): Required<Graph> => {
  const invocationNodes = nodes.filter(isInvocationNodeLike);

  const executableNodes = invocationNodes.filter((node) => {
    const nodeType = node.data.type;
    return !isBatchNodeType(nodeType) && !isGeneratorNodeType(nodeType);
  });

  const parsedNodes = executableNodes.reduce<NonNullable<Graph['nodes']>>((nodesAccumulator, node) => {
    const { id, data } = node;
    const { type } = data;

    const nodeTemplate = templates[type];
    if (!nodeTemplate) {
      log.warn({ id, type }, 'Node template not found!');
      return nodesAccumulator;
    }

    const transformedInputs = reduce(
      data.inputs,
      (inputsAccumulator, input, name) => {
        const fieldTemplate = nodeTemplate.inputs[name];
        if (!fieldTemplate) {
          log.warn({ id, name }, 'Field template not found!');
          return inputsAccumulator;
        }
        if (isBoardFieldInputTemplate(fieldTemplate) && isBoardFieldInputInstance(input)) {
          inputsAccumulator[name] = resolveBoardField(input);
        } else {
          inputsAccumulator[name] = input.value;
        }

        return inputsAccumulator;
      },
      {} as Record<Exclude<string, 'id' | 'type'>, unknown>
    );

    transformedInputs['use_cache'] = data.useCache;

    const graphNode = {
      type,
      id,
      ...transformedInputs,
      is_intermediate: data.isIntermediate,
    } as AnyInvocation;

    Object.assign(nodesAccumulator, {
      [id]: graphNode,
    });

    return nodesAccumulator;
  }, {});

  const executableNodeIds = executableNodes.map(({ id }) => id);

  const parsedEdges = edges
    .filter((edge) => edge.type !== 'collapsed')
    .filter((edge) => executableNodeIds.includes(edge.source) && executableNodeIds.includes(edge.target))
    .reduce<NonNullable<Graph['edges']>>((edgesAccumulator, edge) => {
      const { source, target, sourceHandle, targetHandle } = edge;

      if (!sourceHandle || !targetHandle) {
        log.warn({ source, target, sourceHandle, targetHandle }, 'Missing source or taget handle for edge');
        return edgesAccumulator;
      }

      edgesAccumulator.push({
        source: {
          node_id: source,
          field: sourceHandle,
        },
        destination: {
          node_id: target,
          field: targetHandle,
        },
      });

      return edgesAccumulator;
    }, []);

  parsedEdges.forEach((edge) => {
    const destinationNode = parsedNodes[edge.destination.node_id];
    if (!destinationNode) {
      return;
    }
    const field = edge.destination.field;
    parsedNodes[edge.destination.node_id] = omit(destinationNode, field) as AnyInvocation;
  });

  return {
    id: graphId ?? uuidv4(),
    nodes: parsedNodes,
    edges: parsedEdges,
  };
};

/**
 * Builds a graph from the node editor state.
 */
export const buildNodesGraph = (state: RootState, templates: Templates): Required<Graph> => {
  const { nodes, edges } = selectNodesSlice(state);

  return buildInvocationGraph({
    nodes,
    edges,
    templates,
    resolveBoardField: (field) => getBoardField(field, state),
  });
};
