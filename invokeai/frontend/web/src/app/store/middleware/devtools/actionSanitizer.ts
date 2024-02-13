import type { UnknownAction } from '@reduxjs/toolkit';
import { isAnyGraphBuilt } from 'features/nodes/store/actions';
import { nodeTemplatesBuilt } from 'features/nodes/store/nodesSlice';
import { cloneDeep } from 'lodash-es';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import type { Graph } from 'services/api/types';
import { socketGeneratorProgress } from 'services/events/actions';

export const actionSanitizer = <A extends UnknownAction>(action: A): A => {
  if (isAnyGraphBuilt(action)) {
    if (action.payload.nodes) {
      const sanitizedNodes: Graph['nodes'] = {};

      return {
        ...action,
        payload: { ...action.payload, nodes: sanitizedNodes },
      };
    }
  }

  if (appInfoApi.endpoints.getOpenAPISchema.matchFulfilled(action)) {
    return {
      ...action,
      payload: '<OpenAPI schema omitted>',
    };
  }

  if (nodeTemplatesBuilt.match(action)) {
    return {
      ...action,
      payload: '<Node templates omitted>',
    };
  }

  if (socketGeneratorProgress.match(action)) {
    const sanitized = cloneDeep(action);
    if (sanitized.payload.data.progress_image) {
      sanitized.payload.data.progress_image.dataURL = '<Progress image omitted>';
    }
    return sanitized;
  }

  return action;
};
