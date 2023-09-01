import { NodesState } from 'features/nodes/store/types';
import { InputFieldValue, isInvocationNode } from 'features/nodes/types/types';
import { cloneDeep, omit, reduce } from 'lodash-es';
import { Graph } from 'services/api/types';
import { AnyInvocation } from 'services/events/types';
import { v4 as uuidv4 } from 'uuid';
import { buildWorkflow } from '../buildWorkflow';

/**
 * We need to do special handling for some fields
 */
export const parseFieldValue = (field: InputFieldValue) => {
  if (field.type === 'ColorField') {
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

  return field.value;
};

/**
 * Builds a graph from the node editor state.
 */
export const buildNodesGraph = (nodesState: NodesState): Graph => {
  const { nodes, edges } = nodesState;

  const filteredNodes = nodes.filter(isInvocationNode);
  const workflowJSON = JSON.stringify(buildWorkflow(nodesState));

  // Reduce the node editor nodes into invocation graph nodes
  const parsedNodes = filteredNodes.reduce<NonNullable<Graph['nodes']>>(
    (nodesAccumulator, node) => {
      const { id, data } = node;
      const { type, inputs, isIntermediate, embedWorkflow } = data;

      // Transform each node's inputs to simple key-value pairs
      const transformedInputs = reduce(
        inputs,
        (inputsAccumulator, input, name) => {
          const parsedValue = parseFieldValue(input);
          inputsAccumulator[name] = parsedValue;

          return inputsAccumulator;
        },
        {} as Record<Exclude<string, 'id' | 'type'>, unknown>
      );

      // Build this specific node
      const graphNode = {
        type,
        id,
        ...transformedInputs,
        is_intermediate: isIntermediate,
      };

      if (embedWorkflow) {
        // add the workflow to the node
        Object.assign(graphNode, { workflow: workflowJSON });
      }

      // Add it to the nodes object
      Object.assign(nodesAccumulator, {
        [id]: graphNode,
      });

      return nodesAccumulator;
    },
    {}
  );

  // skip out the "dummy" edges between collapsed nodes
  const filteredEdges = edges.filter((n) => n.type !== 'collapsed');

  // Reduce the node editor edges into invocation graph edges
  const parsedEdges = filteredEdges.reduce<NonNullable<Graph['edges']>>(
    (edgesAccumulator, edge) => {
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
