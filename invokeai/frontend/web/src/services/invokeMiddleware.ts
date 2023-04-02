import { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { emitSubscribe } from 'app/nodesSocketio/actions';
import { AppDispatch, RootState } from 'app/store';
import { setSessionId } from './apiSlice';
import { invokeSession } from './thunks/session';

export const invokeMiddleware: Middleware =
  (store: MiddlewareAPI<AppDispatch, RootState>) => (next) => (action) => {
    const { dispatch } = store;

    if (action.type === 'api/createSession/fulfilled' && action?.payload?.id) {
      const sessionId = action.payload.id;
      console.log('createSession.fulfilled');

      dispatch(setSessionId(sessionId));
      dispatch(emitSubscribe(sessionId));
      dispatch(invokeSession({ sessionId }));
    } else {
      next(action);
    }
  };
