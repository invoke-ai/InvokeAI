import { createAction } from '@reduxjs/toolkit';

/**
 * We can't use redux-toolkit's createSlice() to make these actions,
 * because they have no associated reducer. They only exist to dispatch
 * requests to the server via socketio. These actions will be handled
 * by the middleware.
 */

export const emitSubscribe = createAction<string>('socketio/subscribe');
export const emitUnsubscribe = createAction<string>('socketio/unsubscribe');
