import { AnyAction } from '@reduxjs/toolkit';
import { isAnyGraphBuilt } from 'features/nodes/store/actions';
import { forEach } from 'lodash-es';
import { Graph } from 'services/api';

export const actionSanitizer = <A extends AnyAction>(action: A): A => {
  if (isAnyGraphBuilt(action)) {
    if (action.payload.nodes) {
      const sanitizedNodes: Graph['nodes'] = {};

      // Sanitize nodes as needed
      forEach(action.payload.nodes, (node, key) => {
        // Don't log the whole freaking dataURL
        if (node.type === 'dataURL_image') {
          const { dataURL, ...rest } = node;
          sanitizedNodes[key] = { ...rest, dataURL: '<dataURL>' };
        } else {
          sanitizedNodes[key] = { ...node };
        }
      });

      return {
        ...action,
        payload: { ...action.payload, nodes: sanitizedNodes },
      };
    }
  }

  return action;
};
