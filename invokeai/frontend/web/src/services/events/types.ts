import type { S } from 'services/api/types';
import type { Socket } from 'socket.io-client';

type ClientEmitSubscribeQueue = { queue_id: string };
type ClientEmitUnsubscribeQueue = ClientEmitSubscribeQueue;
type ClientEmitSubscribeBulkDownload = { bulk_download_id: string };
type ClientEmitUnsubscribeBulkDownload = ClientEmitSubscribeBulkDownload;

export type ServerToClientEvents = {
  invocation_progress: (payload: S['InvocationProgressEvent']) => void;
  invocation_complete: (payload: S['InvocationCompleteEvent']) => void;
  invocation_error: (payload: S['InvocationErrorEvent']) => void;
  invocation_started: (payload: S['InvocationStartedEvent']) => void;
  download_started: (payload: S['DownloadStartedEvent']) => void;
  download_progress: (payload: S['DownloadProgressEvent']) => void;
  download_complete: (payload: S['DownloadCompleteEvent']) => void;
  download_cancelled: (payload: S['DownloadCancelledEvent']) => void;
  download_error: (payload: S['DownloadErrorEvent']) => void;
  model_load_started: (payload: S['ModelLoadStartedEvent']) => void;
  model_install_started: (payload: S['ModelInstallStartedEvent']) => void;
  model_install_download_started: (payload: S['ModelInstallDownloadStartedEvent']) => void;
  model_install_download_progress: (payload: S['ModelInstallDownloadProgressEvent']) => void;
  model_install_downloads_complete: (payload: S['ModelInstallDownloadsCompleteEvent']) => void;
  model_install_complete: (payload: S['ModelInstallCompleteEvent']) => void;
  model_install_error: (payload: S['ModelInstallErrorEvent']) => void;
  model_install_cancelled: (payload: S['ModelInstallCancelledEvent']) => void;
  model_load_complete: (payload: S['ModelLoadCompleteEvent']) => void;
  queue_item_status_changed: (payload: S['QueueItemStatusChangedEvent']) => void;
  queue_cleared: (payload: S['QueueClearedEvent']) => void;
  batch_enqueued: (payload: S['BatchEnqueuedEvent']) => void;
  bulk_download_started: (payload: S['BulkDownloadStartedEvent']) => void;
  bulk_download_complete: (payload: S['BulkDownloadCompleteEvent']) => void;
  bulk_download_error: (payload: S['BulkDownloadErrorEvent']) => void;
};

export type ClientToServerEvents = {
  connect: () => void;
  disconnect: () => void;
  subscribe_queue: (payload: ClientEmitSubscribeQueue) => void;
  unsubscribe_queue: (payload: ClientEmitUnsubscribeQueue) => void;
  subscribe_bulk_download: (payload: ClientEmitSubscribeBulkDownload) => void;
  unsubscribe_bulk_download: (payload: ClientEmitUnsubscribeBulkDownload) => void;
};

export type AppSocket = Socket<ServerToClientEvents, ClientToServerEvents>;
