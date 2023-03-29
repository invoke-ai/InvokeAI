import { Middleware } from '@reduxjs/toolkit';
import { setSessionId } from './apiSlice';
import { invokeSession } from './thunks/session';

export const invokeMiddleware: Middleware = (store) => (next) => (action) => {
  const { dispatch } = store;

  if (action.type === 'api/createSession/fulfilled' && action?.payload?.id) {
    console.log('createSession.fulfilled');

    dispatch(setSessionId(action.payload.id));
    // types are wrong but this works
    dispatch(invokeSession({ sessionId: action.payload.id }));
  } else {
    next(action);
  }
};
