import { createAction } from '@reduxjs/toolkit';
import {
  GeneratorProgressEvent,
  InvocationCompleteEvent,
  InvocationErrorEvent,
  InvocationStartedEvent,
} from 'services/events/types';

type SocketioPayload = {
  timestamp: Date;
};

export const socketioConnected = createAction<SocketioPayload>(
  'socketio/socketioConnected'
);

export const socketioDisconnected = createAction<SocketioPayload>(
  'socketio/socketioDisconnected'
);

export const socketioSubscribed = createAction<
  SocketioPayload & { sessionId: string }
>('socketio/socketioSubscribed');

export const socketioUnsubscribed = createAction<
  SocketioPayload & { sessionId: string }
>('socketio/socketioUnsubscribed');

export const invocationStarted = createAction<
  SocketioPayload & { data: InvocationStartedEvent }
>('socketio/invocationStarted');

export const invocationComplete = createAction<
  SocketioPayload & { data: InvocationCompleteEvent }
>('socketio/invocationComplete');

export const invocationError = createAction<
  SocketioPayload & { data: InvocationErrorEvent }
>('socketio/invocationError');

export const generatorProgress = createAction<
  SocketioPayload & { data: GeneratorProgressEvent }
>('socketio/generatorProgress');
