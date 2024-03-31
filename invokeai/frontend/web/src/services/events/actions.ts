import { createAction } from '@reduxjs/toolkit';
import type {
  BulkDownloadCompleteEvent,
  BulkDownloadFailedEvent,
  BulkDownloadStartedEvent,
  DownloadCancelledEvent,
  DownloadCompleteEvent,
  DownloadErrorEvent,
  DownloadProgressEvent,
  DownloadStartedEvent,
  InvocationCompleteEvent,
  InvocationDenoiseProgressEvent,
  InvocationErrorEvent,
  InvocationStartedEvent,
  ModelInstallCancelledEvent,
  ModelInstallCompleteEvent,
  ModelInstallDownloadProgressEvent,
  ModelInstallDownloadsCompleteEvent,
  ModelInstallErrorEvent,
  ModelInstallStartedEvent,
  ModelLoadCompleteEvent,
  ModelLoadStartedEvent,
  QueueItemStatusChangedEvent,
  SessionCanceledEvent,
  SessionCompleteEvent,
  SessionStartedEvent,
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

export const socketSessionStarted = createAction<{
  data: SessionStartedEvent;
}>('socket/socketSessionStarted');

export const socketGraphExecutionStateComplete = createAction<{
  data: SessionCompleteEvent;
}>('socket/socketGraphExecutionStateComplete');

export const socketSessionCanceled = createAction<{
  data: SessionCanceledEvent;
}>('socket/socketSessionCanceled');

export const socketGeneratorProgress = createAction<{
  data: InvocationDenoiseProgressEvent;
}>('socket/socketGeneratorProgress');

export const socketModelLoadStarted = createAction<{
  data: ModelLoadStartedEvent;
}>('socket/socketModelLoadStarted');

export const socketModelLoadComplete = createAction<{
  data: ModelLoadCompleteEvent;
}>('socket/socketModelLoadComplete');

export const socketDownloadStarted = createAction<{
  data: DownloadStartedEvent;
}>('socket/socketDownloadStarted');

export const socketDownloadProgress = createAction<{
  data: DownloadProgressEvent;
}>('socket/socketDownloadProgress');

export const socketDownloadComplete = createAction<{
  data: DownloadCompleteEvent;
}>('socket/socketDownloadComplete');

export const socketDownloadCancelled = createAction<{
  data: DownloadCancelledEvent;
}>('socket/socketDownloadCancelled');

export const socketDownloadError = createAction<{
  data: DownloadErrorEvent;
}>('socket/socketDownloadError');

export const socketModelInstallStarted = createAction<{
  data: ModelInstallStartedEvent;
}>('socket/socketModelInstallStarted');

export const socketModelInstallDownloadProgress = createAction<{
  data: ModelInstallDownloadProgressEvent;
}>('socket/socketModelInstallDownloadProgress');

export const socketModelInstallDownloadsComplete = createAction<{
  data: ModelInstallDownloadsCompleteEvent;
}>('socket/socketModelInstallDownloadsComplete');

export const socketModelInstallComplete = createAction<{
  data: ModelInstallCompleteEvent;
}>('socket/socketModelInstallComplete');

export const socketModelInstallError = createAction<{
  data: ModelInstallErrorEvent;
}>('socket/socketModelInstallError');

export const socketModelInstallCancelled = createAction<{
  data: ModelInstallCancelledEvent;
}>('socket/socketModelInstallCancelled');

export const socketQueueItemStatusChanged = createAction<{
  data: QueueItemStatusChangedEvent;
}>('socket/socketQueueItemStatusChanged');

export const socketBulkDownloadStarted = createAction<{
  data: BulkDownloadStartedEvent;
}>('socket/socketBulkDownloadStarted');

export const socketBulkDownloadComplete = createAction<{
  data: BulkDownloadCompleteEvent;
}>('socket/socketBulkDownloadComplete');

export const socketBulkDownloadError = createAction<{
  data: BulkDownloadFailedEvent;
}>('socket/socketBulkDownloadError');
