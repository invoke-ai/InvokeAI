import { RootState } from 'app/store/store';
import { InputFieldValue } from 'features/nodes/types/types';
import { cloneDeep, omit, reduce } from 'lodash-es';
import { Graph } from 'services/api/types';
import { AnyInvocation } from 'services/events/types';
import { v4 as uuidv4 } from 'uuid';
import { modelIdToLoRAModelField } from '../modelIdToLoRAName';
import { modelIdToMainModelField } from '../modelIdToMainModelField';
import { modelIdToVAEModelField } from '../modelIdToVAEModelField';

/**
 * We need to do special handling for some fields
 */
export const parseFieldValue = (field: InputFieldValue) => {
  if (field.type === 'color') {
    if (field.value) {
      const clonedValue = cloneDeep(field.value);

      const { r, g, b, a } = field.value;

      // scale alpha value to PIL's desired range 0-255
      const scaledAlpha = Math.max(0, Math.min(a * 255, 255));
      const transformedColor = { r, g, b, a: scaledAlpha };

      Object.assign(clonedValue, transformedColor);
      return clonedValue;
    }
  }

  if (field.type === 'model') {
    if (field.value) {
      return modelIdToMainModelField(field.value);
    }
  }

  if (field.type === 'vae_model') {
    if (field.value) {
      return modelIdToVAEModelField(field.value);
    }
  }

  if (field.type === 'lora_model') {
    if (field.value) {
      return modelIdToLoRAModelField(field.value);
    }
  }

  return field.value;
};

/**
 * Builds a graph from the node editor state.
 */
export const buildNodesGraph = (state: RootState): Graph => {
  const { nodes, edges } = state.nodes;

  const filteredNodes = nodes.filter((n) => n.type !== 'progress_image');

  // Reduce the node editor nodes into invocation graph nodes
  const parsedNodes = filteredNodes.reduce<NonNullable<Graph['nodes']>>(
    (nodesAccumulator, node, nodeIndex) => {
      const { id, data } = node;
      const { type, inputs } = data;

      // Transform each node's inputs to simple key-value pairs
      const transformedInputs = reduce(
        inputs,
        (inputsAccumulator, input, name) => {
          const parsedValue = parseFieldValue(input);
          inputsAccumulator[name] = parsedValue;

          return inputsAccumulator;
        },
        {} as Record<Exclude<string, 'id' | 'type'>, any>
      );

      // Build this specific node
      const graphNode = {
        type,
        id,
        ...transformedInputs,
      };

      // Add it to the nodes object
      Object.assign(nodesAccumulator, {
        [id]: graphNode,
      });

      return nodesAccumulator;
    },
    {}
  );

  // Reduce the node editor edges into invocation graph edges
  const parsedEdges = edges.reduce<NonNullable<Graph['edges']>>(
    (edgesAccumulator, edge, edgeIndex) => {
      const { source, target, sourceHandle, targetHandle } = edge;

      // Format the edges and add to the edges array
      edgesAccumulator.push({
        source: {
          node_id: source,
          field: sourceHandle as string,
        },
        destination: {
          node_id: target,
          field: targetHandle as string,
        },
      });

      return edgesAccumulator;
    },
    []
  );

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
    parsedNodes[edge.destination.node_id] = omit(
      destination_node,
      field
    ) as AnyInvocation;
  });

  // Assemble!
  const graph = {
    id: uuidv4(),
    nodes: parsedNodes,
    edges: parsedEdges,
  };

  return graph;
};
