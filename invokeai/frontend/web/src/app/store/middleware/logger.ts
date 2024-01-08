import type { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { diff } from 'deep-object-diff';

/**
 * Super simple logger middleware. Useful for debugging when the redux devtools are awkward.
 */
export const debuggingLoggerMiddleware: Middleware =
  (api: MiddlewareAPI) => (next) => (action) => {
    const originalState = api.getState();
    console.debug('dispatching', action);
    const result = next(action);
    const nextState = api.getState();
    const stateDiff = diff(originalState, nextState);
    console.debug('diff', stateDiff);
    return result;
  };
