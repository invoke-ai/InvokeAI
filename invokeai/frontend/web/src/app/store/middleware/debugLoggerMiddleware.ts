import type { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { diff } from 'jsondiffpatch';

/**
 * Super simple logger middleware. Useful for debugging when the redux devtools are awkward.
 */
export const debugLoggerMiddleware: Middleware = (api: MiddlewareAPI) => (next) => (action) => {
  const originalState = api.getState();
  console.log('REDUX: dispatching', action);
  const result = next(action);
  const nextState = api.getState();
  console.log('REDUX: next state', nextState);
  console.log('REDUX: diff', diff(originalState, nextState));
  return result;
};
