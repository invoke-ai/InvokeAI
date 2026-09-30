import type { UnknownAction } from '@reduxjs/toolkit';
import { appInfoApi } from 'services/api/endpoints/appInfo';

export const actionSanitizer = <A extends UnknownAction>(action: A): A => {
  if (appInfoApi.endpoints.getOpenAPISchema.matchFulfilled(action)) {
    return {
      ...action,
      payload: '<OpenAPI schema omitted>',
    };
  }

  return action;
};
