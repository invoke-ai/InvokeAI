import { ExternalLink } from '@invoke-ai/ui-library';
import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { getIsCancelled } from 'app/store/middleware/listenerMiddleware/listeners/cancellationsListeners';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { $bulkDownloadId } from 'app/store/nanostores/bulkDownloadId';
import { $queueId } from 'app/store/nanostores/queueId';
import type { AppDispatch, RootState } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { $lastCanvasProgressEvent } from 'features/controlLayers/store/canvasSlice';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import ErrorToastDescription, { getTitleFromErrorType } from 'features/toast/ErrorToastDescription';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { forEach } from 'lodash-es';
import { atom, computed } from 'nanostores';
import { api, LIST_TAG } from 'services/api';
import { modelsApi } from 'services/api/endpoints/models';
import { queueApi, queueItemsAdapter } from 'services/api/endpoints/queue';
import type { S } from 'services/api/types';
import { buildOnInvocationComplete } from 'services/events/onInvocationComplete';
import type { ClientToServerEvents, ServerToClientEvents } from 'services/events/types';
import type { Socket } from 'socket.io-client';

export const socketConnected = createAction('socket/connected');

const log = logger('events');

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  dispatch: AppDispatch;
  getState: () => RootState;
  setIsConnected: (isConnected: boolean) => void;
};

const selectModelInstalls = modelsApi.endpoints.listModelInstalls.select();
const nodeTypeDenylist = ['load_image', 'image'];
export const $lastProgressEvent = atom<S['InvocationDenoiseProgressEvent'] | null>(null);
export const $hasProgress = computed($lastProgressEvent, (val) => Boolean(val));
export const $progressImage = computed($lastProgressEvent, (val) => val?.progress_image ?? null);
export const $isProgressFromCanvas = computed($lastProgressEvent, (val) => val?.destination === 'canvas');

export const setEventListeners = ({ socket, dispatch, getState, setIsConnected }: SetEventListenersArg) => {
  socket.on('connect', () => {
    log.debug('Connected');
    setIsConnected(true);
    dispatch(socketConnected());
    const queue_id = $queueId.get();
    socket.emit('subscribe_queue', { queue_id });
    if (!$baseUrl.get()) {
      const bulk_download_id = $bulkDownloadId.get();
      socket.emit('subscribe_bulk_download', { bulk_download_id });
    }
    $lastProgressEvent.set(null);
    $lastCanvasProgressEvent.set(null);
  });

  socket.on('connect_error', (error) => {
    log.debug('Connect error');
    setIsConnected(false);
    $lastProgressEvent.set(null);
    $lastCanvasProgressEvent.set(null);
    if (error && error.message) {
      const data: string | undefined = (error as unknown as { data: string | undefined }).data;
      if (data === 'ERR_UNAUTHENTICATED') {
        toast({
          id: `connect-error-${error.message}`,
          title: error.message,
          status: 'error',
          duration: 10000,
        });
      }
    }
  });

  socket.on('disconnect', () => {
    log.debug('Disconnected');
    $lastProgressEvent.set(null);
    $lastCanvasProgressEvent.set(null);
    setIsConnected(false);
  });

  socket.on('invocation_started', (data) => {
    const { invocation_source_id, invocation } = data;
    log.debug({ data } as SerializableObject, `Invocation started (${invocation.type}, ${invocation_source_id})`);
    const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
    if (nes) {
      nes.status = zNodeStatus.enum.IN_PROGRESS;
      upsertExecutionState(nes.nodeId, nes);
    }
  });

  socket.on('invocation_denoise_progress', (data) => {
    const {
      invocation_source_id,
      invocation,
      step,
      total_steps,
      progress_image,
      origin,
      destination,
      percentage,
      session_id,
      batch_id,
    } = data;

    if (getIsCancelled({ session_id, batch_id, destination })) {
      // Do not update the progress if this session has been cancelled. This prevents a race condition where we get a
      // progress update after the session has been cancelled.
      return;
    }

    log.trace(
      { data } as SerializableObject,
      `Denoise ${Math.round(percentage * 100)}% (${invocation.type}, ${invocation_source_id})`
    );

    $lastProgressEvent.set(data);

    if (origin === 'workflows') {
      const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
      if (nes) {
        nes.status = zNodeStatus.enum.IN_PROGRESS;
        nes.progress = (step + 1) / total_steps;
        nes.progressImage = progress_image ?? null;
        upsertExecutionState(nes.nodeId, nes);
      }
    }

    // This event is only relevant for the canvas
    if (destination === 'canvas') {
      $lastCanvasProgressEvent.set(data);
    }
  });

  socket.on('invocation_error', (data) => {
    const { invocation_source_id, invocation, error_type, error_message, error_traceback } = data;
    log.error({ data } as SerializableObject, `Invocation error (${invocation.type}, ${invocation_source_id})`);
    const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
    if (nes) {
      nes.status = zNodeStatus.enum.FAILED;
      nes.progress = null;
      nes.progressImage = null;
      nes.error = {
        error_type,
        error_message,
        error_traceback,
      };
      upsertExecutionState(nes.nodeId, nes);
    }
  });

  const onInvocationComplete = buildOnInvocationComplete(
    getState,
    dispatch,
    nodeTypeDenylist,
    $lastProgressEvent.set,
    $lastCanvasProgressEvent.set
  );
  socket.on('invocation_complete', onInvocationComplete);

  socket.on('model_load_started', (data) => {
    const { config, submodel_type } = data;
    const { name, base, type } = config;

    const extras: string[] = [base, type];

    if (submodel_type) {
      extras.push(submodel_type);
    }

    const message = `Model load started: ${name} (${extras.join(', ')})`;

    log.debug({ data }, message);
  });

  socket.on('model_load_complete', (data) => {
    const { config, submodel_type } = data;
    const { name, base, type } = config;

    const extras: string[] = [base, type];
    if (submodel_type) {
      extras.push(submodel_type);
    }

    const message = `Model load complete: ${name} (${extras.join(', ')})`;

    log.debug({ data }, message);
  });

  socket.on('download_started', (data) => {
    log.debug({ data }, 'Download started');
  });

  socket.on('download_progress', (data) => {
    log.trace({ data }, 'Download progress');
  });

  socket.on('download_complete', (data) => {
    log.debug({ data }, 'Download complete');
  });

  socket.on('download_cancelled', (data) => {
    log.warn({ data }, 'Download cancelled');
  });

  socket.on('download_error', (data) => {
    log.error({ data }, 'Download error');
  });

  socket.on('model_install_started', (data) => {
    log.debug({ data }, 'Model install started');

    const { id } = data;
    const installs = selectModelInstalls(getState()).data;

    if (!installs?.find((install) => install.id === id)) {
      dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
    } else {
      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'running';
          }
          return draft;
        })
      );
    }
  });

  socket.on('model_install_download_started', (data) => {
    log.debug({ data }, 'Model install download started');

    const { id } = data;
    const installs = selectModelInstalls(getState()).data;

    if (!installs?.find((install) => install.id === id)) {
      dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
    } else {
      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'downloading';
          }
          return draft;
        })
      );
    }
  });

  socket.on('model_install_download_progress', (data) => {
    log.trace({ data }, 'Model install download progress');

    const { bytes, total_bytes, id } = data;
    const installs = selectModelInstalls(getState()).data;

    if (!installs?.find((install) => install.id === id)) {
      dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
    } else {
      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.bytes = bytes;
            modelImport.total_bytes = total_bytes;
            modelImport.status = 'downloading';
          }
          return draft;
        })
      );
    }
  });

  socket.on('model_install_downloads_complete', (data) => {
    log.debug({ data }, 'Model install downloads complete');

    const { id } = data;
    const installs = selectModelInstalls(getState()).data;

    if (!installs?.find((install) => install.id === id)) {
      dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
    } else {
      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'downloads_done';
          }
          return draft;
        })
      );
    }
  });

  socket.on('model_install_complete', (data) => {
    log.debug({ data }, 'Model install complete');

    const { id } = data;

    const installs = selectModelInstalls(getState()).data;

    if (!installs?.find((install) => install.id === id)) {
      dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
    } else {
      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'completed';
          }
          return draft;
        })
      );
    }

    dispatch(api.util.invalidateTags([{ type: 'ModelConfig', id: LIST_TAG }]));
    dispatch(api.util.invalidateTags([{ type: 'ModelScanFolderResults', id: LIST_TAG }]));
  });

  socket.on('model_install_error', (data) => {
    log.error({ data }, 'Model install error');

    const { id, error, error_type } = data;
    const installs = selectModelInstalls(getState()).data;

    if (!installs?.find((install) => install.id === id)) {
      dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
    } else {
      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'error';
            modelImport.error_reason = error_type;
            modelImport.error = error;
          }
          return draft;
        })
      );
    }
  });

  socket.on('model_install_cancelled', (data) => {
    log.warn({ data }, 'Model install cancelled');

    const { id } = data;
    const installs = selectModelInstalls(getState()).data;

    if (!installs?.find((install) => install.id === id)) {
      dispatch(api.util.invalidateTags([{ type: 'ModelInstalls' }]));
    } else {
      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'cancelled';
          }
          return draft;
        })
      );
    }
  });

  socket.on('queue_item_status_changed', (data) => {
    // we've got new status for the queue item, batch and queue
    const {
      item_id,
      session_id,
      status,
      started_at,
      updated_at,
      completed_at,
      batch_status,
      queue_status,
      error_type,
      error_message,
      error_traceback,
      origin,
    } = data;

    log.debug({ data }, `Queue item ${item_id} status updated: ${status}`);

    // Update this specific queue item in the list of queue items (this is the queue item DTO, without the session)
    dispatch(
      queueApi.util.updateQueryData('listQueueItems', undefined, (draft) => {
        queueItemsAdapter.updateOne(draft, {
          id: String(item_id),
          changes: {
            status,
            started_at,
            updated_at: updated_at ?? undefined,
            completed_at: completed_at ?? undefined,
            error_type,
            error_message,
            error_traceback,
          },
        });
      })
    );

    // Update the queue status (we do not get the processor status here)
    dispatch(
      queueApi.util.updateQueryData('getQueueStatus', undefined, (draft) => {
        if (!draft) {
          return;
        }
        Object.assign(draft.queue, queue_status);
      })
    );

    // Update the batch status
    dispatch(queueApi.util.updateQueryData('getBatchStatus', { batch_id: batch_status.batch_id }, () => batch_status));

    // Invalidate caches for things we cannot update
    // TODO: technically, we could possibly update the current session queue item, but feels safer to just request it again
    dispatch(
      queueApi.util.invalidateTags([
        'CurrentSessionQueueItem',
        'NextSessionQueueItem',
        'InvocationCacheStatus',
        { type: 'SessionQueueItem', id: item_id },
      ])
    );

    if (status === 'in_progress') {
      forEach($nodeExecutionStates.get(), (nes) => {
        if (!nes) {
          return;
        }
        const clone = deepClone(nes);
        clone.status = zNodeStatus.enum.PENDING;
        clone.error = null;
        clone.progress = null;
        clone.progressImage = null;
        clone.outputs = [];
        $nodeExecutionStates.setKey(clone.nodeId, clone);
      });
    } else if (status === 'failed' && error_type) {
      const isLocal = getState().config.isLocal ?? true;
      const sessionId = session_id;
      $lastProgressEvent.set(null);

      if (origin === 'canvas') {
        $lastCanvasProgressEvent.set(null);
      }

      toast({
        id: `INVOCATION_ERROR_${error_type}`,
        title: getTitleFromErrorType(error_type),
        status: 'error',
        duration: null,
        updateDescription: isLocal,
        description: (
          <ErrorToastDescription
            errorType={error_type}
            errorMessage={error_message}
            sessionId={sessionId}
            isLocal={isLocal}
          />
        ),
      });
    } else if (status === 'canceled') {
      $lastProgressEvent.set(null);
      if (origin === 'canvas') {
        $lastCanvasProgressEvent.set(null);
      }
    } else if (status === 'completed') {
      $lastProgressEvent.set(null);
    }
  });

  socket.on('queue_cleared', (data) => {
    log.debug({ data }, 'Queue cleared');
  });

  socket.on('batch_enqueued', (data) => {
    log.debug({ data }, 'Batch enqueued');
  });

  socket.on('bulk_download_started', (data) => {
    log.debug({ data }, 'Bulk gallery download preparation started');
  });

  socket.on('bulk_download_complete', (data) => {
    log.debug({ data }, 'Bulk gallery download ready');
    const { bulk_download_item_name } = data;

    // TODO(psyche): This URL may break in in some environments (e.g. Nvidia workbench) but we need to test it first
    const url = `/api/v1/images/download/${bulk_download_item_name}`;

    toast({
      id: bulk_download_item_name,
      title: t('gallery.bulkDownloadReady', 'Download ready'),
      status: 'success',
      description: (
        <ExternalLink
          label={t('gallery.clickToDownload', 'Click here to download')}
          href={url}
          download={bulk_download_item_name}
        />
      ),
      duration: null,
    });
  });

  socket.on('bulk_download_error', (data) => {
    log.error({ data }, 'Bulk gallery download error');

    const { bulk_download_item_name, error } = data;

    toast({
      id: bulk_download_item_name,
      title: t('gallery.bulkDownloadFailed'),
      status: 'error',
      description: error,
      duration: null,
    });
  });
};
