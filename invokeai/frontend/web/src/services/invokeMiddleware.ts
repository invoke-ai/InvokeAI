import { isFulfilled, Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { emitSubscribe } from 'app/nodesSocketio/actions';
import { AppDispatch, RootState } from 'app/store';
import { setSessionId } from './apiSlice';
import { createSession, invokeSession } from './thunks/session';

/**
 * `redux-toolkit` provides nice matching utilities, which can be used as type guards
 * See: https://redux-toolkit.js.org/api/matching-utilities
 */

const isFulfilledCreateSession = isFulfilled(createSession);

export const invokeMiddleware: Middleware =
  (store: MiddlewareAPI<AppDispatch, RootState>) => (next) => (action) => {
    const { dispatch } = store;

    if (isFulfilledCreateSession(action)) {
      const sessionId = action.payload.id;
      console.log('createSession.fulfilled');

      dispatch(setSessionId(sessionId));
      dispatch(emitSubscribe(sessionId));
      dispatch(invokeSession({ sessionId }));
    } else {
      next(action);
    }
  };
