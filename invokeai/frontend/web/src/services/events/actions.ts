import { createAction } from '@reduxjs/toolkit';
import {
  GeneratorProgressEvent,
  GraphExecutionStateCompleteEvent,
  InvocationCompleteEvent,
  InvocationErrorEvent,
  InvocationStartedEvent,
} from 'services/events/types';

// Common socket action payload data
type BaseSocketPayload = {
  timestamp: string;
};

// Create actions for each socket
// Middleware and redux can then respond to them as needed

/**
 * Socket.IO Connected
 *
 * Do not use. Only for use in middleware.
 */
export const socketConnected = createAction<BaseSocketPayload>(
  'socket/socketConnected'
);

/**
 * App-level Socket.IO Connected
 */
export const appSocketConnected = createAction<BaseSocketPayload>(
  'socket/appSocketConnected'
);

/**
 * Socket.IO Disconnect
 *
 * Do not use. Only for use in middleware.
 */
export const socketDisconnected = createAction<BaseSocketPayload>(
  'socket/socketDisconnected'
);

/**
 * App-level Socket.IO Disconnected
 */
export const appSocketDisconnected = createAction<BaseSocketPayload>(
  'socket/appSocketDisconnected'
);

/**
 * Socket.IO Subscribed
 *
 * Do not use. Only for use in middleware.
 */
export const socketSubscribed = createAction<
  BaseSocketPayload & { sessionId: string }
>('socket/socketSubscribed');

/**
 * App-level Socket.IO Subscribed
 */
export const appSocketSubscribed = createAction<
  BaseSocketPayload & { sessionId: string }
>('socket/appSocketSubscribed');

/**
 * Socket.IO Unsubscribed
 *
 * Do not use. Only for use in middleware.
 */
export const socketUnsubscribed = createAction<
  BaseSocketPayload & { sessionId: string }
>('socket/socketUnsubscribed');

/**
 * App-level Socket.IO Unsubscribed
 */
export const appSocketUnsubscribed = createAction<
  BaseSocketPayload & { sessionId: string }
>('socket/appSocketUnsubscribed');

/**
 * Socket.IO Invocation Started
 *
 * Do not use. Only for use in middleware.
 */
export const socketInvocationStarted = createAction<
  BaseSocketPayload & { data: InvocationStartedEvent }
>('socket/socketInvocationStarted');

/**
 * App-level Socket.IO Invocation Started
 */
export const appSocketInvocationStarted = createAction<
  BaseSocketPayload & { data: InvocationStartedEvent }
>('socket/appSocketInvocationStarted');

/**
 * Socket.IO Invocation Complete
 *
 * Do not use. Only for use in middleware.
 */
export const socketInvocationComplete = createAction<
  BaseSocketPayload & {
    data: InvocationCompleteEvent;
  }
>('socket/socketInvocationComplete');

/**
 * App-level Socket.IO Invocation Complete
 */
export const appSocketInvocationComplete = createAction<
  BaseSocketPayload & {
    data: InvocationCompleteEvent;
  }
>('socket/appSocketInvocationComplete');

/**
 * Socket.IO Invocation Error
 *
 * Do not use. Only for use in middleware.
 */
export const socketInvocationError = createAction<
  BaseSocketPayload & { data: InvocationErrorEvent }
>('socket/socketInvocationError');

/**
 * App-level Socket.IO Invocation Error
 */
export const appSocketInvocationError = createAction<
  BaseSocketPayload & { data: InvocationErrorEvent }
>('socket/appSocketInvocationError');

/**
 * Socket.IO Graph Execution State Complete
 *
 * Do not use. Only for use in middleware.
 */
export const socketGraphExecutionStateComplete = createAction<
  BaseSocketPayload & { data: GraphExecutionStateCompleteEvent }
>('socket/socketGraphExecutionStateComplete');

/**
 * App-level Socket.IO Graph Execution State Complete
 */
export const appSocketGraphExecutionStateComplete = createAction<
  BaseSocketPayload & { data: GraphExecutionStateCompleteEvent }
>('socket/appSocketGraphExecutionStateComplete');

/**
 * Socket.IO Generator Progress
 *
 * Do not use. Only for use in middleware.
 */
export const socketGeneratorProgress = createAction<
  BaseSocketPayload & { data: GeneratorProgressEvent }
>('socket/socketGeneratorProgress');

/**
 * App-level Socket.IO Generator Progress
 */
export const appSocketGeneratorProgress = createAction<
  BaseSocketPayload & { data: GeneratorProgressEvent }
>('socket/appSocketGeneratorProgress');
