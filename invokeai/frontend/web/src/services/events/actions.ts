import { createAction } from '@reduxjs/toolkit';
import type {
  BulkDownloadCompletedEvent,
  BulkDownloadFailedEvent,
  BulkDownloadStartedEvent,
  GeneratorProgressEvent,
  GraphExecutionStateCompleteEvent,
  InvocationCompleteEvent,
  InvocationErrorEvent,
  InvocationRetrievalErrorEvent,
  InvocationStartedEvent,
  ModelLoadCompletedEvent,
  ModelLoadStartedEvent,
  ModelInstallDownloadingEvent,
  ModelInstallCompletedEvent,
  ModelInstallErrorEvent,
  QueueItemStatusChangedEvent,
  SessionRetrievalErrorEvent,
} from 'services/events/types';

// Create actions for each socket
// Middleware and redux can then respond to them as needed

export const socketConnected = createAction('socket/socketConnected');

export const socketDisconnected = createAction('socket/socketDisconnected');

export const socketSubscribedSession = createAction<{
  sessionId: string;
}>('socket/socketSubscribedSession');

export const socketUnsubscribedSession = createAction<{ sessionId: string }>('socket/socketUnsubscribedSession');

export const socketInvocationStarted = createAction<{
  data: InvocationStartedEvent;
}>('socket/socketInvocationStarted');

export const socketInvocationComplete = createAction<{
  data: InvocationCompleteEvent;
}>('socket/socketInvocationComplete');

export const socketInvocationError = createAction<{
  data: InvocationErrorEvent;
}>('socket/socketInvocationError');

export const socketGraphExecutionStateComplete = createAction<{
  data: GraphExecutionStateCompleteEvent;
}>('socket/socketGraphExecutionStateComplete');

export const socketGeneratorProgress = createAction<{
  data: GeneratorProgressEvent;
}>('socket/socketGeneratorProgress');

export const socketModelLoadStarted = createAction<{
  data: ModelLoadStartedEvent;
}>('socket/socketModelLoadStarted');

export const socketModelLoadCompleted = createAction<{
  data: ModelLoadCompletedEvent;
}>('socket/socketModelLoadCompleted');

export const socketModelInstallDownloading = createAction<{
  data: ModelInstallDownloadingEvent;
}>('socket/socketModelInstallDownloading');

export const socketModelInstallCompleted = createAction<{
  data: ModelInstallCompletedEvent;
}>('socket/socketModelInstallCompleted');

export const socketModelInstallError = createAction<{
  data: ModelInstallErrorEvent;
}>('socket/socketModelInstallError');

export const socketSessionRetrievalError = createAction<{
  data: SessionRetrievalErrorEvent;
}>('socket/socketSessionRetrievalError');

export const socketInvocationRetrievalError = createAction<{
  data: InvocationRetrievalErrorEvent;
}>('socket/socketInvocationRetrievalError');

export const socketQueueItemStatusChanged = createAction<{
  data: QueueItemStatusChangedEvent;
}>('socket/socketQueueItemStatusChanged');

export const socketBulkDownloadStarted = createAction<{
  data: BulkDownloadStartedEvent;
}>('socket/socketBulkDownloadStarted');

export const socketBulkDownloadCompleted = createAction<{
  data: BulkDownloadCompletedEvent;
}>('socket/socketBulkDownloadCompleted');

export const socketBulkDownloadFailed = createAction<{
  data: BulkDownloadFailedEvent;
}>('socket/socketBulkDownloadFailed');
