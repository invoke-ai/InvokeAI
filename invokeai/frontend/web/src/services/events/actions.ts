import { createAction } from '@reduxjs/toolkit';
import type {
  GeneratorProgressEvent,
  GraphExecutionStateCompleteEvent,
  InvocationCompleteEvent,
  InvocationErrorEvent,
  InvocationRetrievalErrorEvent,
  InvocationStartedEvent,
  ModelLoadCompletedEvent,
  ModelLoadStartedEvent,
  QueueItemStatusChangedEvent,
  SessionRetrievalErrorEvent,
} from 'services/events/types';

// Create actions for each socket
// Middleware and redux can then respond to them as needed

/**
 * Socket.IO Connected
 *
 * Do not use. Only for use in middleware.
 */
export const socketConnected = createAction('socket/socketConnected');

/**
 * App-level Socket.IO Connected
 */
export const appSocketConnected = createAction('socket/appSocketConnected');

/**
 * Socket.IO Disconnect
 *
 * Do not use. Only for use in middleware.
 */
export const socketDisconnected = createAction('socket/socketDisconnected');

/**
 * App-level Socket.IO Disconnected
 */
export const appSocketDisconnected = createAction(
  'socket/appSocketDisconnected'
);

/**
 * Socket.IO Subscribed Session
 *
 * Do not use. Only for use in middleware.
 */
export const socketSubscribedSession = createAction<{
  sessionId: string;
}>('socket/socketSubscribedSession');

/**
 * App-level Socket.IO Subscribed Session
 */
export const appSocketSubscribedSession = createAction<{
  sessionId: string;
}>('socket/appSocketSubscribedSession');

/**
 * Socket.IO Unsubscribed Session
 *
 * Do not use. Only for use in middleware.
 */
export const socketUnsubscribedSession = createAction<{ sessionId: string }>(
  'socket/socketUnsubscribedSession'
);

/**
 * App-level Socket.IO Unsubscribed Session
 */
export const appSocketUnsubscribedSession = createAction<{ sessionId: string }>(
  'socket/appSocketUnsubscribedSession'
);

/**
 * Socket.IO Invocation Started
 *
 * Do not use. Only for use in middleware.
 */
export const socketInvocationStarted = createAction<{
  data: InvocationStartedEvent;
}>('socket/socketInvocationStarted');

/**
 * App-level Socket.IO Invocation Started
 */
export const appSocketInvocationStarted = createAction<{
  data: InvocationStartedEvent;
}>('socket/appSocketInvocationStarted');

/**
 * Socket.IO Invocation Complete
 *
 * Do not use. Only for use in middleware.
 */
export const socketInvocationComplete = createAction<{
  data: InvocationCompleteEvent;
}>('socket/socketInvocationComplete');

/**
 * App-level Socket.IO Invocation Complete
 */
export const appSocketInvocationComplete = createAction<{
  data: InvocationCompleteEvent;
}>('socket/appSocketInvocationComplete');

/**
 * Socket.IO Invocation Error
 *
 * Do not use. Only for use in middleware.
 */
export const socketInvocationError = createAction<{
  data: InvocationErrorEvent;
}>('socket/socketInvocationError');

/**
 * App-level Socket.IO Invocation Error
 */
export const appSocketInvocationError = createAction<{
  data: InvocationErrorEvent;
}>('socket/appSocketInvocationError');

/**
 * Socket.IO Graph Execution State Complete
 *
 * Do not use. Only for use in middleware.
 */
export const socketGraphExecutionStateComplete = createAction<{
  data: GraphExecutionStateCompleteEvent;
}>('socket/socketGraphExecutionStateComplete');

/**
 * App-level Socket.IO Graph Execution State Complete
 */
export const appSocketGraphExecutionStateComplete = createAction<{
  data: GraphExecutionStateCompleteEvent;
}>('socket/appSocketGraphExecutionStateComplete');

/**
 * Socket.IO Generator Progress
 *
 * Do not use. Only for use in middleware.
 */
export const socketGeneratorProgress = createAction<{
  data: GeneratorProgressEvent;
}>('socket/socketGeneratorProgress');

/**
 * App-level Socket.IO Generator Progress
 */
export const appSocketGeneratorProgress = createAction<{
  data: GeneratorProgressEvent;
}>('socket/appSocketGeneratorProgress');

/**
 * Socket.IO Model Load Started
 *
 * Do not use. Only for use in middleware.
 */
export const socketModelLoadStarted = createAction<{
  data: ModelLoadStartedEvent;
}>('socket/socketModelLoadStarted');

/**
 * App-level Model Load Started
 */
export const appSocketModelLoadStarted = createAction<{
  data: ModelLoadStartedEvent;
}>('socket/appSocketModelLoadStarted');

/**
 * Socket.IO Model Load Started
 *
 * Do not use. Only for use in middleware.
 */
export const socketModelLoadCompleted = createAction<{
  data: ModelLoadCompletedEvent;
}>('socket/socketModelLoadCompleted');

/**
 * App-level Model Load Completed
 */
export const appSocketModelLoadCompleted = createAction<{
  data: ModelLoadCompletedEvent;
}>('socket/appSocketModelLoadCompleted');

/**
 * Socket.IO Session Retrieval Error
 *
 * Do not use. Only for use in middleware.
 */
export const socketSessionRetrievalError = createAction<{
  data: SessionRetrievalErrorEvent;
}>('socket/socketSessionRetrievalError');

/**
 * App-level Session Retrieval Error
 */
export const appSocketSessionRetrievalError = createAction<{
  data: SessionRetrievalErrorEvent;
}>('socket/appSocketSessionRetrievalError');

/**
 * Socket.IO Invocation Retrieval Error
 *
 * Do not use. Only for use in middleware.
 */
export const socketInvocationRetrievalError = createAction<{
  data: InvocationRetrievalErrorEvent;
}>('socket/socketInvocationRetrievalError');

/**
 * App-level Invocation Retrieval Error
 */
export const appSocketInvocationRetrievalError = createAction<{
  data: InvocationRetrievalErrorEvent;
}>('socket/appSocketInvocationRetrievalError');

/**
 * Socket.IO Quueue Item Status Changed
 *
 * Do not use. Only for use in middleware.
 */
export const socketQueueItemStatusChanged = createAction<{
  data: QueueItemStatusChangedEvent;
}>('socket/socketQueueItemStatusChanged');

/**
 * App-level Quueue Item Status Changed
 */
export const appSocketQueueItemStatusChanged = createAction<{
  data: QueueItemStatusChangedEvent;
}>('socket/appSocketQueueItemStatusChanged');
