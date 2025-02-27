import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import type { BoardField } from 'features/nodes/types/common';
import type { BoardFieldInputInstance } from 'features/nodes/types/field';
import { isBoardFieldInputInstance, isBoardFieldInputTemplate } from 'features/nodes/types/field';
import { isExecutableNode, isInvocationNode } from 'features/nodes/types/invocation';
import { omit, reduce } from 'lodash-es';
import type { AnyInvocation, Graph } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

const log = logger('workflows');

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

/**
 * Builds a graph from the node editor state.
 */
export const buildNodesGraph = (state: RootState, templates: Templates): Graph => {
  const { nodes, edges } = selectNodesSlice(state);

  // Exclude all batch nodes - we will handle these in the batch setup in a diff function
  const filteredNodes = nodes.filter(isInvocationNode).filter(isExecutableNode);

  // Reduce the node editor nodes into invocation graph nodes
  const parsedNodes = filteredNodes.reduce<NonNullable<Graph['nodes']>>((nodesAccumulator, node) => {
    const { id, data } = node;
    const { type, inputs, isIntermediate } = data;

    const nodeTemplate = templates[type];
    if (!nodeTemplate) {
      log.warn({ id, type }, 'Node template not found!');
      return nodesAccumulator;
    }

    // Transform each node's inputs to simple key-value pairs
    const transformedInputs = reduce(
      inputs,
      (inputsAccumulator, input, name) => {
        const fieldTemplate = nodeTemplate.inputs[name];
        if (!fieldTemplate) {
          log.warn({ id, name }, 'Field template not found!');
          return inputsAccumulator;
        }
        if (isBoardFieldInputTemplate(fieldTemplate) && isBoardFieldInputInstance(input)) {
          inputsAccumulator[name] = getBoardField(input, state);
        } else {
          inputsAccumulator[name] = input.value;
        }

        return inputsAccumulator;
      },
      {} as Record<Exclude<string, 'id' | 'type'>, unknown>
    );

    // add reserved use_cache
    transformedInputs['use_cache'] = node.data.useCache;

    // Build this specific node
    const graphNode = {
      type,
      id,
      ...transformedInputs,
      is_intermediate: isIntermediate,
    };

    // Add it to the nodes object
    Object.assign(nodesAccumulator, {
      [id]: graphNode,
    });

    return nodesAccumulator;
  }, {});

  const filteredNodeIds = filteredNodes.map(({ id }) => id);

  // skip out the "dummy" edges between collapsed nodes
  const filteredEdges = edges
    .filter((edge) => edge.type !== 'collapsed')
    .filter((edge) => filteredNodeIds.includes(edge.source) && filteredNodeIds.includes(edge.target));

  // Reduce the node editor edges into invocation graph edges
  const parsedEdges = filteredEdges.reduce<NonNullable<Graph['edges']>>((edgesAccumulator, edge) => {
    const { source, target, sourceHandle, targetHandle } = edge;

    if (!sourceHandle || !targetHandle) {
      log.warn({ source, target, sourceHandle, targetHandle }, 'Missing source or taget handle for edge');
      return edgesAccumulator;
    }

    // Format the edges and add to the edges array
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

  /**
   * Omit all inputs that have edges connected.
   *
   * Fixes edge case where the user has connected an input, but also provided an invalid explicit,
   * value.
   *
   * In this edge case, pydantic will invalidate the node based on the invalid explicit value,
   * even though the actual value that will be used comes from the connection.
   */
  parsedEdges.forEach((edge) => {
    const destination_node = parsedNodes[edge.destination.node_id];
    const field = edge.destination.field;
    parsedNodes[edge.destination.node_id] = omit(destination_node, field) as AnyInvocation;
  });

  // Assemble!
  const graph = {
    id: uuidv4(),
    nodes: parsedNodes,
    edges: parsedEdges,
  };

  return graph;
};
