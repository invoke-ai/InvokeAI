/* eslint-disable no-console */
// This is only enabled manually for debugging, console is allowed.

import type { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { diff } from 'jsondiffpatch';

/**
 * Super simple logger middleware. Useful for debugging when the redux devtools are awkward.
 */
export const getDebugLoggerMiddleware =
  (options?: { withDiff?: boolean; withNextState?: boolean }): Middleware =>
  (api: MiddlewareAPI) =>
  (next) =>
  (action) => {
    const originalState = api.getState();
    console.log('REDUX: dispatching', action);
    const result = next(action);
    const nextState = api.getState();
    if (options?.withNextState) {
      console.log('REDUX: next state', nextState);
    }
    if (options?.withDiff) {
      console.log('REDUX: diff', diff(originalState, nextState));
    }
    return result;
  };
