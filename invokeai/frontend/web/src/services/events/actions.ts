import { createAction } from '@reduxjs/toolkit';
import type {
  BatchEnqueuedEvent,
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
  ModelInstallDownloadStartedEvent,
  ModelInstallErrorEvent,
  ModelInstallStartedEvent,
  ModelLoadCompleteEvent,
  ModelLoadStartedEvent,
  QueueClearedEvent,
  QueueItemStatusChangedEvent,
} from 'services/events/types';

const createSocketAction = <T = undefined>(name: string) =>
  createAction<T extends undefined ? void : { data: T }>(`socket/${name}`);

export const socketConnected = createSocketAction('Connected');
export const socketDisconnected = createSocketAction('Disconnected');
export const socketInvocationStarted = createSocketAction<InvocationStartedEvent>('InvocationStartedEvent');
export const socketInvocationComplete = createSocketAction<InvocationCompleteEvent>('InvocationCompleteEvent');
export const socketInvocationError = createSocketAction<InvocationErrorEvent>('InvocationErrorEvent');
export const socketGeneratorProgress = createSocketAction<InvocationDenoiseProgressEvent>(
  'InvocationDenoiseProgressEvent'
);
export const socketModelLoadStarted = createSocketAction<ModelLoadStartedEvent>('ModelLoadStartedEvent');
export const socketModelLoadComplete = createSocketAction<ModelLoadCompleteEvent>('ModelLoadCompleteEvent');
export const socketDownloadStarted = createSocketAction<DownloadStartedEvent>('DownloadStartedEvent');
export const socketDownloadProgress = createSocketAction<DownloadProgressEvent>('DownloadProgressEvent');
export const socketDownloadComplete = createSocketAction<DownloadCompleteEvent>('DownloadCompleteEvent');
export const socketDownloadCancelled = createSocketAction<DownloadCancelledEvent>('DownloadCancelledEvent');
export const socketDownloadError = createSocketAction<DownloadErrorEvent>('DownloadErrorEvent');
export const socketModelInstallStarted = createSocketAction<ModelInstallStartedEvent>('ModelInstallStartedEvent');
export const socketModelInstallDownloadProgress = createSocketAction<ModelInstallDownloadProgressEvent>(
  'ModelInstallDownloadProgressEvent'
);
export const socketModelInstallDownloadStarted = createSocketAction<ModelInstallDownloadStartedEvent>(
  'ModelInstallDownloadStartedEvent'
);
export const socketModelInstallDownloadsComplete = createSocketAction<ModelInstallDownloadsCompleteEvent>(
  'ModelInstallDownloadsCompleteEvent'
);
export const socketModelInstallComplete = createSocketAction<ModelInstallCompleteEvent>('ModelInstallCompleteEvent');
export const socketModelInstallError = createSocketAction<ModelInstallErrorEvent>('ModelInstallErrorEvent');
export const socketModelInstallCancelled = createSocketAction<ModelInstallCancelledEvent>('ModelInstallCancelledEvent');
export const socketQueueItemStatusChanged =
  createSocketAction<QueueItemStatusChangedEvent>('QueueItemStatusChangedEvent');
export const socketQueueCleared = createSocketAction<QueueClearedEvent>('QueueClearedEvent');
export const socketBatchEnqueued = createSocketAction<BatchEnqueuedEvent>('BatchEnqueuedEvent');
export const socketBulkDownloadStarted = createSocketAction<BulkDownloadStartedEvent>('BulkDownloadStartedEvent');
export const socketBulkDownloadComplete = createSocketAction<BulkDownloadCompleteEvent>('BulkDownloadCompleteEvent');
export const socketBulkDownloadError = createSocketAction<BulkDownloadFailedEvent>('BulkDownloadFailedEvent');
