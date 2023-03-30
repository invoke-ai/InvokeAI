import { Middleware } from '@reduxjs/toolkit';
import { emitSubscribe } from 'app/nodesSocketio/actions';
import { setSessionId } from './apiSlice';
import { invokeSession } from './thunks/session';

export const invokeMiddleware: Middleware = (store) => (next) => (action) => {
  const { dispatch } = store;

  if (action.type === 'api/createSession/fulfilled' && action?.payload?.id) {
    const sessionId = action.payload.id;
    console.log('createSession.fulfilled');

    dispatch(setSessionId(sessionId));
    dispatch(emitSubscribe(sessionId));
    // types are wrong but this works
    dispatch(invokeSession({ sessionId }));
  } else {
    next(action);
  }
};
