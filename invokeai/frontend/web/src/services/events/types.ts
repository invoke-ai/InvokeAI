import type { S } from 'services/api/types';

type ModelLoadStartedEvent = S['ModelLoadStartedEvent'];
type ModelLoadCompleteEvent = S['ModelLoadCompleteEvent'];

type InvocationStartedEvent = S['InvocationStartedEvent'];
type InvocationDenoiseProgressEvent = S['InvocationDenoiseProgressEvent'];
type InvocationCompleteEvent = S['InvocationCompleteEvent'];
type InvocationErrorEvent = S['InvocationErrorEvent'];

type ModelInstallDownloadStartedEvent = S['ModelInstallDownloadStartedEvent'];
type ModelInstallDownloadProgressEvent = S['ModelInstallDownloadProgressEvent'];
type ModelInstallDownloadsCompleteEvent = S['ModelInstallDownloadsCompleteEvent'];
type ModelInstallCompleteEvent = S['ModelInstallCompleteEvent'];
type ModelInstallErrorEvent = S['ModelInstallErrorEvent'];
type ModelInstallStartedEvent = S['ModelInstallStartedEvent'];
type ModelInstallCancelledEvent = S['ModelInstallCancelledEvent'];

type DownloadStartedEvent = S['DownloadStartedEvent'];
type DownloadProgressEvent = S['DownloadProgressEvent'];
type DownloadCompleteEvent = S['DownloadCompleteEvent'];
type DownloadCancelledEvent = S['DownloadCancelledEvent'];
type DownloadErrorEvent = S['DownloadErrorEvent'];

type QueueItemStatusChangedEvent = S['QueueItemStatusChangedEvent'];
type QueueClearedEvent = S['QueueClearedEvent'];
type BatchEnqueuedEvent = S['BatchEnqueuedEvent'];

type BulkDownloadStartedEvent = S['BulkDownloadStartedEvent'];
type BulkDownloadCompleteEvent = S['BulkDownloadCompleteEvent'];
type BulkDownloadFailedEvent = S['BulkDownloadErrorEvent'];

type ClientEmitSubscribeQueue = {
  queue_id: string;
};
type ClientEmitUnsubscribeQueue = ClientEmitSubscribeQueue;
type ClientEmitSubscribeBulkDownload = {
  bulk_download_id: string;
};
type ClientEmitUnsubscribeBulkDownload = ClientEmitSubscribeBulkDownload;

export type ServerToClientEvents = {
  invocation_denoise_progress: (payload: InvocationDenoiseProgressEvent) => void;
  invocation_complete: (payload: InvocationCompleteEvent) => void;
  invocation_error: (payload: InvocationErrorEvent) => void;
  invocation_started: (payload: InvocationStartedEvent) => void;
  download_started: (payload: DownloadStartedEvent) => void;
  download_progress: (payload: DownloadProgressEvent) => void;
  download_complete: (payload: DownloadCompleteEvent) => void;
  download_cancelled: (payload: DownloadCancelledEvent) => void;
  download_error: (payload: DownloadErrorEvent) => void;
  model_load_started: (payload: ModelLoadStartedEvent) => void;
  model_install_started: (payload: ModelInstallStartedEvent) => void;
  model_install_download_started: (payload: ModelInstallDownloadStartedEvent) => void;
  model_install_download_progress: (payload: ModelInstallDownloadProgressEvent) => void;
  model_install_downloads_complete: (payload: ModelInstallDownloadsCompleteEvent) => void;
  model_install_complete: (payload: ModelInstallCompleteEvent) => void;
  model_install_error: (payload: ModelInstallErrorEvent) => void;
  model_install_cancelled: (payload: ModelInstallCancelledEvent) => void;
  model_load_complete: (payload: ModelLoadCompleteEvent) => void;
  queue_item_status_changed: (payload: QueueItemStatusChangedEvent) => void;
  queue_cleared: (payload: QueueClearedEvent) => void;
  batch_enqueued: (payload: BatchEnqueuedEvent) => void;
  bulk_download_started: (payload: BulkDownloadStartedEvent) => void;
  bulk_download_complete: (payload: BulkDownloadCompleteEvent) => void;
  bulk_download_error: (payload: BulkDownloadFailedEvent) => void;
};

export type ClientToServerEvents = {
  connect: () => void;
  disconnect: () => void;
  subscribe_queue: (payload: ClientEmitSubscribeQueue) => void;
  unsubscribe_queue: (payload: ClientEmitUnsubscribeQueue) => void;
  subscribe_bulk_download: (payload: ClientEmitSubscribeBulkDownload) => void;
  unsubscribe_bulk_download: (payload: ClientEmitUnsubscribeBulkDownload) => void;
};
