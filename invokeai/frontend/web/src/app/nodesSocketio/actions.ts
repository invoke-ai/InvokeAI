import { createAction } from '@reduxjs/toolkit';
import {
  GeneratorProgressEvent,
  InvocationCompleteEvent,
  InvocationErrorEvent,
  InvocationStartedEvent,
} from 'services/events/types';
/**
 * We can't use redux-toolkit's createSlice() to make these actions,
 * because they have no associated reducer. They only exist to dispatch
 * requests to the server via socketio. These actions will be handled
 * by the middleware.
 */

export const emitSubscribe = createAction<string>('socketio/subscribe');
export const emitUnsubscribe = createAction<string>('socketio/unsubscribe');

type Timestamp = {
  timestamp: Date;
};

export const invocationStarted = createAction<
  { data: InvocationStartedEvent } & Timestamp
>('socketio/invocationStarted');

export const invocationComplete = createAction<
  { data: InvocationCompleteEvent } & Timestamp
>('socketio/invocationComplete');

export const invocationError = createAction<
  { data: InvocationErrorEvent } & Timestamp
>('socketio/invocationError');

export const generatorProgress = createAction<
  { data: GeneratorProgressEvent } & Timestamp
>('socketio/generatorProgress');
