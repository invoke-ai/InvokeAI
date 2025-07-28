import { ExternalLink } from '@invoke-ai/ui-library';
import { isAnyOf } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { socketConnected } from 'app/store/middleware/listenerMiddleware/listeners/socketConnected';
import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { $bulkDownloadId } from 'app/store/nanostores/bulkDownloadId';
import { $queueId } from 'app/store/nanostores/queueId';
import type { AppStore } from 'app/store/store';
import { listenerMiddleware } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { forEach, isNil, round } from 'es-toolkit/compat';
import {
  $isInPublishFlow,
  $outputNodeId,
  $validationRunData,
} from 'features/nodes/components/sidePanel/workflow/publish';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { zNodeStatus } from 'features/nodes/types/invocation';
import ErrorToastDescription, { getTitle } from 'features/toast/ErrorToastDescription';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { LRUCache } from 'lru-cache';
import type { ApiTagDescription } from 'services/api';
import { api, LIST_ALL_TAG, LIST_TAG } from 'services/api';
import { modelsApi } from 'services/api/endpoints/models';
import { queueApi, queueItemsAdapter } from 'services/api/endpoints/queue';
import { workflowsApi } from 'services/api/endpoints/workflows';
import { buildOnInvocationComplete } from 'services/events/onInvocationComplete';
import { buildOnModelInstallError } from 'services/events/onModelInstallError';
import type { ClientToServerEvents, ServerToClientEvents } from 'services/events/types';
import type { Socket } from 'socket.io-client';
import type { JsonObject } from 'type-fest';

import { $lastProgressEvent } from './stores';

const log = logger('events');

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  store: AppStore;
  setIsConnected: (isConnected: boolean) => void;
};

const selectModelInstalls = modelsApi.endpoints.listModelInstalls.select();

export const setEventListeners = ({ socket, store, setIsConnected }: SetEventListenersArg) => {
  const { dispatch, getState } = store;

  const finishedQueueItemIds = new LRUCache<number, boolean>({ max: 100 });

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
  });

  socket.on('connect_error', (error) => {
    log.debug('Connect error');
    setIsConnected(false);
    $lastProgressEvent.set(null);
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
    setIsConnected(false);
  });

  socket.on('invocation_started', (data) => {
    if (finishedQueueItemIds.has(data.item_id)) {
      return;
    }
    const { invocation_source_id, invocation } = data;
    log.debug({ data } as JsonObject, `Invocation started (${invocation.type}, ${invocation_source_id})`);
    const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
    if (nes) {
      nes.status = zNodeStatus.enum.IN_PROGRESS;
      upsertExecutionState(nes.nodeId, nes);
    }
  });

  socket.on('invocation_progress', (data) => {
    if (finishedQueueItemIds.has(data.item_id)) {
      log.trace({ data } as JsonObject, `Received event for already-finished queue item ${data.item_id}`);
      return;
    }
    const { invocation_source_id, invocation, image, origin, percentage, message } = data;

    let _message = 'Invocation progress';
    if (message) {
      _message += `: ${message}`;
    }
    if (!isNil(percentage)) {
      _message += ` ${round(percentage * 100, 2)}%`;
    }
    _message += ` (${invocation.type}, ${invocation_source_id})`;

    log.trace({ data } as JsonObject, _message);

    $lastProgressEvent.set(data);

    if (origin === 'workflows') {
      const nes = deepClone($nodeExecutionStates.get()[invocation_source_id]);
      if (nes) {
        nes.status = zNodeStatus.enum.IN_PROGRESS;
        nes.progress = percentage;
        nes.progressImage = image ?? null;
        upsertExecutionState(nes.nodeId, nes);
      }
    }
  });

  socket.on('invocation_error', (data) => {
    if (finishedQueueItemIds.has(data.item_id)) {
      log.trace({ data } as JsonObject, `Received event for already-finished queue item ${data.item_id}`);
      return;
    }
    const { invocation_source_id, invocation, error_type, error_message, error_traceback } = data;
    log.error({ data } as JsonObject, `Invocation error (${invocation.type}, ${invocation_source_id})`);
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

  const onInvocationComplete = buildOnInvocationComplete(getState, dispatch, finishedQueueItemIds);
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

  const onModelInstallError = buildOnModelInstallError(getState, dispatch);
  socket.on('model_install_error', onModelInstallError);

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
    if (finishedQueueItemIds.has(data.item_id)) {
      log.trace({ data }, `Received event for already-finished queue item ${data.item_id}`);
      return;
    }

    // we've got new status for the queue item, batch and queue
    const {
      item_id,
      session_id,
      status,
      batch_status,
      error_type,
      error_message,
      destination,
      started_at,
      updated_at,
      completed_at,
      error_traceback,
      credits,
    } = data;

    log.debug({ data }, `Queue item ${item_id} status updated: ${status}`);

    // // Update this specific queue item in the list of queue items
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
            credits,
          },
        });
      })
    );

    // Invalidate caches for things we cannot easily update
    const tagsToInvalidate: ApiTagDescription[] = [
      'CurrentSessionQueueItem',
      'NextSessionQueueItem',
      'InvocationCacheStatus',
      { type: 'SessionQueueItem', id: item_id },
      { type: 'SessionQueueItem', id: LIST_TAG },
      { type: 'SessionQueueItem', id: LIST_ALL_TAG },
      { type: 'BatchStatus', id: batch_status.batch_id },
    ];
    if (destination) {
      tagsToInvalidate.push({ type: 'QueueCountsByDestination', id: destination });
    }
    dispatch(queueApi.util.invalidateTags(tagsToInvalidate));
    dispatch(
      queueApi.util.updateQueryData('getQueueStatus', undefined, (draft) => {
        draft.queue = data.queue_status;
      })
    );
    dispatch(
      queueApi.util.updateQueryData('getBatchStatus', { batch_id: data.batch_id }, (draft) => {
        Object.assign(draft, data.batch_status);
      })
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
    } else if (status === 'completed' || status === 'failed' || status === 'canceled') {
      finishedQueueItemIds.set(item_id, true);
      if (status === 'failed' && error_type) {
        const isLocal = getState().config.isLocal ?? true;
        const sessionId = session_id;

        toast({
          id: `INVOCATION_ERROR_${error_type}`,
          title: getTitle(error_type),
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
      }
      // If the queue item is completed, failed, or cancelled, we want to clear the last progress event
      $lastProgressEvent.set(null);
      // $progressImages.setKey(session_id, undefined);

      // When a validation run is completed, we want to clear the validation run batch ID & set the workflow as published
      const validationRunData = $validationRunData.get();
      if (!validationRunData || batch_status.batch_id !== validationRunData.batchId || status !== 'completed') {
        return;
      }

      // The published status of a workflow is server state, provided to the client in by the getWorkflow query.
      // After successfully publishing a workflow, we need to invalidate the query cache so that the published status is
      // seen throughout the app. We also need to reset the publish flow state.
      //
      // But, there is a race condition! If we invalidate the query cache and then immediately clear the publish flow state,
      // between the time when the publish state is cleared and the query is re-fetched, we will render the wrong UI.
      //
      // So, we really need to wait for the query re-fetch to complete before clearing the publish flow state. This isn't
      // possible using the `invalidateTags()` API. But we can fudge it by adding a once-off listener for that query.

      listenerMiddleware.startListening({
        matcher: isAnyOf(
          workflowsApi.endpoints.getWorkflow.matchFulfilled,
          workflowsApi.endpoints.getWorkflow.matchRejected
        ),
        effect: (action, listenerApi) => {
          if (workflowsApi.endpoints.getWorkflow.matchFulfilled(action)) {
            // If this query was re-fetching the workflow that was just published, we can clear the publish flow state and
            // unsubscribe from the listener
            if (action.payload.workflow_id === validationRunData.workflowId) {
              listenerApi.unsubscribe();
              $validationRunData.set(null);
              $isInPublishFlow.set(false);
              $outputNodeId.set(null);
            }
          } else if (workflowsApi.endpoints.getWorkflow.matchRejected(action)) {
            // If the query failed, we can unsubscribe from the listener
            listenerApi.unsubscribe();
          }
        },
      });
      dispatch(workflowsApi.util.invalidateTags([{ type: 'Workflow', id: validationRunData.workflowId }]));
    }
  });

  socket.on('queue_cleared', (data) => {
    log.debug({ data }, 'Queue cleared');
  });

  socket.on('batch_enqueued', (data) => {
    log.debug({ data }, 'Batch enqueued');
  });

  socket.on('queue_items_retried', (data) => {
    log.debug({ data }, 'Queue items retried');
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
