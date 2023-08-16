import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
// import { validateSeedWeights } from 'common/util/seedWeightPairs';
import { every } from 'lodash-es';
import { getConnectedEdges } from 'reactflow';
import { isInvocationNode } from '../types/types';
import { NodesState } from './types';

export const selectIsReadyNodes = createSelector(
  [stateSelector],
  (state) => {
    const { nodes, system } = state;
    const { isProcessing, isConnected } = system;

    if (isProcessing || !isConnected) {
      // Cannot generate if already processing an image
      return false;
    }

    if (!nodes.shouldValidateGraph) {
      return true;
    }

    const isGraphReady = every(nodes.nodes, (node) => {
      if (!isInvocationNode(node)) {
        return true;
      }

      const nodeTemplate = nodes.nodeTemplates[node.data.type];

      if (!nodeTemplate) {
        // Node type not found
        return false;
      }

      const connectedEdges = getConnectedEdges([node], nodes.edges);

      const isNodeValid = every(node.data.inputs, (field) => {
        const fieldTemplate = nodeTemplate.inputs[field.name];
        const hasConnection = connectedEdges.some(
          (edge) => edge.target === node.id && edge.targetHandle === field.name
        );

        if (!fieldTemplate) {
          // Field type not found
          return false;
        }

        if (fieldTemplate.required && !field.value && !hasConnection) {
          // Required field is empty or does not have a connection
          return false;
        }

        // ok
        return true;
      });

      return isNodeValid;
    });

    return isGraphReady;
  },
  defaultSelectorOptions
);

export const getNodeAndTemplate = (nodeId: string, nodes: NodesState) => {
  const node = nodes.nodes.find((node) => node.id === nodeId);
  const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];

  return { node, nodeTemplate };
};

export const getInputFieldAndTemplate = (
  nodeId: string,
  fieldName: string,
  nodes: NodesState
) => {
  const node = nodes.nodes
    .filter(isInvocationNode)
    .find((node) => node.id === nodeId);
  const nodeTemplate = nodes.nodeTemplates[node?.data.type ?? ''];

  if (!node || !nodeTemplate) {
    return;
  }

  const field = node.data.inputs[fieldName];
  const fieldTemplate = nodeTemplate.inputs[fieldName];

  return { field, fieldTemplate };
};
