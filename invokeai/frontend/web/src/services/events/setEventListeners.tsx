import { Flex, Text } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { socketConnected } from 'app/store/middleware/listenerMiddleware/listeners/socketConnected';
import type { AppStore } from 'app/store/store';
import { parseify } from 'common/util/serialize';
import { isNil, round } from 'es-toolkit/compat';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { allEntitiesDeleted, controlLayerRecalled } from 'features/controlLayers/store/canvasSlice';
import { canvasWorkflowIntegrationProcessingCompleted } from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import { loraAllDeleted, loraRecalled } from 'features/controlLayers/store/lorasSlice';
import {
  heightChanged,
  negativePromptChanged,
  positivePromptChanged,
  setCfgScale,
  setSeed,
  setSteps,
  widthChanged,
} from 'features/controlLayers/store/paramsSlice';
import { refImagesRecalled } from 'features/controlLayers/store/refImagesSlice';
import type {
  ControlModeV2,
  FLUXReduxImageInfluence,
  IPMethodV2,
  RefImageState,
} from 'features/controlLayers/store/types';
import { getControlLayerState, getReferenceImageState } from 'features/controlLayers/store/util';
import { $nodeExecutionStates, upsertExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { modelSelected } from 'features/parameters/store/actions';
import { toast, toastApi } from 'features/toast/toast';
import { t } from 'i18next';
import { Trans } from 'react-i18next';
import type { ApiTagDescription } from 'services/api';
import { api, LIST_TAG } from 'services/api';
import { imagesApi } from 'services/api/endpoints/images';
import { modelsApi } from 'services/api/endpoints/models';
import { queueApi } from 'services/api/endpoints/queue';
import { getEventScope } from 'services/events/eventScope';
import { buildOnForeignInvocationComplete, buildOnInvocationComplete } from 'services/events/onInvocationComplete';
import { buildOnModelInstallError, DiscordLink, GitHubIssuesLink } from 'services/events/onModelInstallError';
import {
  buildOnNonOwnerQueueItemStatusChanged,
  buildOnQueueItemStatusChanged,
} from 'services/events/onQueueItemStatusChanged';
import { QUEUE_CHANGED_TAGS } from 'services/events/queueCacheTags';
import type { ClientToServerEvents, ServerToClientEvents } from 'services/events/types';
import { createWorkflowExecutionCoordinator } from 'services/events/workflowExecutionCoordinator';
import type { Socket } from 'socket.io-client';
import type { JsonObject } from 'type-fest';

import { $lastProgressEvent, $loadingModelsCount, clearLLMTaskState, setLLMTaskState } from './stores';

const log = logger('events');

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  store: AppStore;
  setIsConnected: (isConnected: boolean) => void;
};

const selectModelInstalls = modelsApi.endpoints.listModelInstalls.select();

/**
 * Sets up event listeners for the socketio client. Some components will set up their own listeners. These are the ones
 * that have app-wide implications.
 */
export const setEventListeners = ({ socket, store, setIsConnected }: SetEventListenersArg) => {
  const { dispatch, getState } = store;

  const completedInvocationKeysByItemId = new Map<number, Set<string>>();
  const onInvocationComplete = buildOnInvocationComplete(getState, dispatch, completedInvocationKeysByItemId);
  const workflowExecutionCoordinator = createWorkflowExecutionCoordinator({
    clearCanvasWorkflowIntegrationProcessing: () => dispatch(canvasWorkflowIntegrationProcessingCompleted()),
    completedInvocationKeysByItemId,
    getAllNodeExecutionStates: () => $nodeExecutionStates.get(),
    getCurrentUserId: () => selectCurrentUser(getState())?.user_id ?? null,
    getNodeExecutionState: (nodeId) => $nodeExecutionStates.get()[nodeId],
    logReconciliationError: (error, itemId) => {
      log.debug({ error: parseify(error) }, `Unable to reconcile workflow queue item ${itemId}`);
    },
    onInvocationComplete,
    reconcileQueueItem: (itemId) =>
      dispatch(queueApi.endpoints.getQueueItem.initiate(itemId, { forceRefetch: true, subscribe: false })),
    setNodeExecutionState: (nodeId, state) => $nodeExecutionStates.setKey(nodeId, state),
    upsertNodeExecutionState: upsertExecutionState,
  });

  const onForeignInvocationComplete = buildOnForeignInvocationComplete(dispatch);
  const onQueueItemStatusChanged = buildOnQueueItemStatusChanged(dispatch, workflowExecutionCoordinator);
  const onNonOwnerQueueItemStatusChanged = buildOnNonOwnerQueueItemStatusChanged(dispatch);

  socket.on('connect', () => {
    log.debug('Connected');
    setIsConnected(true);
    dispatch(socketConnected());
    socket.emit('subscribe_queue', { queue_id: 'default' });
    socket.emit('subscribe_bulk_download', { bulk_download_id: 'default' });
    $lastProgressEvent.set(null);
    $loadingModelsCount.set(0);
  });

  socket.on('connect_error', (error) => {
    log.debug('Connect error');
    setIsConnected(false);
    $lastProgressEvent.set(null);
    $loadingModelsCount.set(0);
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
    workflowExecutionCoordinator.cancelPendingWorkflowReconciliations();
    $lastProgressEvent.set(null);
    $loadingModelsCount.set(0);
    setIsConnected(false);
  });

  const invalidateWorkflowLibrary = () => {
    dispatch(
      api.util.invalidateTags([
        { type: 'Workflow', id: LIST_TAG },
        'WorkflowTags',
        'WorkflowTagCounts',
        'WorkflowCategoryCounts',
      ])
    );
  };

  const clearSavedWorkflowSelection = (workflowId: string) => {
    const nodes = selectNodesSlice(getState()).nodes;

    for (const node of nodes) {
      if (node.type !== 'invocation' || node.data.type !== 'call_saved_workflow') {
        continue;
      }

      if (node.data.inputs.workflow_id?.value !== workflowId) {
        continue;
      }

      dispatch(
        fieldValueReset({
          nodeId: node.id,
          fieldName: 'workflow_id',
          value: '',
        })
      );
    }
  };

  socket.on('workflow_created', (data) => {
    log.debug({ data }, 'Workflow created');
    invalidateWorkflowLibrary();
  });

  socket.on('workflow_updated', (data) => {
    log.debug({ data }, 'Workflow updated');
    invalidateWorkflowLibrary();
  });

  socket.on('workflow_deleted', (data) => {
    log.debug({ data }, 'Workflow deleted');
    invalidateWorkflowLibrary();
    clearSavedWorkflowSelection(data.workflow_id);
  });

  socket.on('workflow_access_revoked', (data) => {
    log.debug({ data }, 'Workflow access revoked');
    invalidateWorkflowLibrary();
    const currentUser = selectCurrentUser(getState());
    if (currentUser?.is_admin || currentUser?.user_id === data.user_id) {
      return;
    }
    clearSavedWorkflowSelection(data.workflow_id);
  });

  // In multiuser mode, admins are subscribed to the "admin" socket room and receive invocation
  // and queue item events for *every* user, carrying that user's real user_id. Another user's
  // events must not drive this client's personal state: the workflow execution coordinator, node
  // execution states, canvas workflow integration processing, completed-invocation bookkeeping,
  // or $lastProgressEvent. Ownership is decided here at the listener — before the coordinator
  // records anything — so each handler only ever sees the events it owns:
  //
  // - Foreign invocation_started/progress/error are dropped. (The backend already routes progress
  //   to the owner's room only; the client-side check is defense in depth.)
  // - A foreign invocation_complete downgrades to a cache-invalidation-only gallery refresh so an
  //   admin viewing another user's board stays fresh without optimistic cache work or DTO fetches.
  // - A non-owner queue_item_status_changed (sanitized companion or admin-room copy) only
  //   invalidates queue tags.
  //
  // In single-user mode there is no authenticated user and every event is 'own'.
  socket.on('invocation_started', (data) => {
    if (getEventScope(getState, data) !== 'own') {
      log.trace({ data } as JsonObject, `Ignoring invocation_started for another user (${data.user_id})`);
      return;
    }
    const { invocation_source_id, invocation } = data;
    log.debug({ data } as JsonObject, `Invocation started (${invocation.type}, ${invocation_source_id})`);
    workflowExecutionCoordinator.onInvocationStarted(data);
  });

  socket.on('invocation_progress', (data) => {
    if (getEventScope(getState, data) !== 'own') {
      log.trace({ data } as JsonObject, `Ignoring invocation_progress for another user (${data.user_id})`);
      return;
    }
    if (!workflowExecutionCoordinator.onInvocationProgress(data)) {
      log.trace({ data } as JsonObject, `Received event for already-finished queue item ${data.item_id}`);
      return;
    }
    const { invocation_source_id, invocation, percentage, message } = data;

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
  });

  socket.on('invocation_error', (data) => {
    if (getEventScope(getState, data) !== 'own') {
      log.trace({ data } as JsonObject, `Ignoring invocation_error for another user (${data.user_id})`);
      return;
    }
    const { invocation_source_id, invocation } = data;
    log.error({ data } as JsonObject, `Invocation error (${invocation.type}, ${invocation_source_id})`);
    workflowExecutionCoordinator.onInvocationError(data);
  });

  socket.on('invocation_complete', (data) => {
    if (getEventScope(getState, data) === 'own') {
      workflowExecutionCoordinator.onInvocationComplete(data);
    } else {
      onForeignInvocationComplete(data);
    }
  });

  socket.on('model_load_started', (data) => {
    const { config, submodel_type } = data;
    const { name, base, type } = config;

    const extras: string[] = [base, type];

    if (submodel_type) {
      extras.push(submodel_type);
    }

    const message = `Model load started: ${name} (${extras.join(', ')})`;

    log.debug({ data }, message);
    $loadingModelsCount.set($loadingModelsCount.get() + 1);
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
    $loadingModelsCount.set(Math.max(0, $loadingModelsCount.get() - 1));
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

    const { id, config } = data;

    if (config.type === 'unknown') {
      toast({
        id: 'UNKNOWN_MODEL',
        title: t('modelManager.unidentifiedModelTitle'),
        description: (
          <Flex flexDir="column" gap={2}>
            <Text fontSize="md" as="span">
              <Trans i18nKey="modelManager.unidentifiedModelMessage" />
            </Text>
            <Text fontSize="md" as="span">
              <Trans
                i18nKey="modelManager.unidentifiedModelMessage2"
                components={{ DiscordLink: <DiscordLink />, GitHubIssuesLink: <GitHubIssuesLink /> }}
              />
            </Text>
          </Flex>
        ),
        status: 'error',
        isClosable: true,
        duration: null,
      });
    }

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
    if (getEventScope(getState, data) === 'own') {
      onQueueItemStatusChanged(data);
    } else {
      onNonOwnerQueueItemStatusChanged(data);
    }
  });

  socket.on('queue_cleared', (data) => {
    log.debug({ data }, 'Queue cleared');
    // Clearing the queue deletes the in-progress item without emitting a per-item terminal status
    // event, so the progress bar must be reset here — and the coordinator must mark the deleted
    // tracked items terminal so a trailing invocation_progress event cannot repopulate the bar.
    // The coordinator scopes a user-scoped clear (multiuser mode) to that user's items — on an
    // admin client that may be a subset of the tracked items; on another user's client it is
    // none of them — and reports whether the clear applied to any tracked item, so the progress
    // bar is only reset when the clear could have deleted the item behind it. The queue tags
    // below always need refreshing.
    if (workflowExecutionCoordinator.onQueueCleared(data)) {
      $lastProgressEvent.set(null);
    }
    dispatch(
      queueApi.util.invalidateTags([
        ...QUEUE_CHANGED_TAGS,
        'SessionProcessorStatus',
        'BatchStatus',
        'QueueCountsByDestination',
      ])
    );
  });

  socket.on('batch_enqueued', (data) => {
    log.debug({ data }, 'Batch enqueued');
    dispatch(queueApi.util.invalidateTags([...QUEUE_CHANGED_TAGS, 'QueueCountsByDestination']));
  });

  // Bulk queue item events (retried/canceled) are the only signal other clients get for a bulk
  // operation, which changes many rows in one SQL statement and emits no per-item
  // queue_item_status_changed. Owners receive their own item ids, admins receive all of them,
  // and other users receive a sanitized companion with no ids — in every case, refetch the
  // queue caches so lists and badge counts update.
  const invalidateQueueTagsForBulkItemEvent = (itemIds: number[]) => {
    const tagsToInvalidate: ApiTagDescription[] = [...QUEUE_CHANGED_TAGS, 'BatchStatus', 'QueueCountsByDestination'];
    // Invalidate each affected item specifically
    for (const itemId of itemIds) {
      tagsToInvalidate.push({ type: 'SessionQueueItem', id: itemId });
    }
    dispatch(queueApi.util.invalidateTags(tagsToInvalidate));
  };

  socket.on('queue_items_retried', (data) => {
    log.debug({ data }, 'Queue items retried');
    invalidateQueueTagsForBulkItemEvent(data.retried_item_ids ?? []);
  });

  socket.on('queue_items_canceled', (data) => {
    log.debug({ data }, 'Queue items canceled');
    invalidateQueueTagsForBulkItemEvent(data.canceled_item_ids);
  });

  socket.on('recall_parameters_updated', (data) => {
    log.debug('Recall parameters updated');

    // Apply the recall parameters to the store
    if (data.parameters) {
      let appliedCount = 0;

      // Map the recall parameter names to store actions
      if (data.parameters.positive_prompt !== undefined && typeof data.parameters.positive_prompt === 'string') {
        dispatch(positivePromptChanged(data.parameters.positive_prompt));
        appliedCount++;
      }
      if (data.parameters.negative_prompt !== undefined && typeof data.parameters.negative_prompt === 'string') {
        dispatch(negativePromptChanged(data.parameters.negative_prompt));
        appliedCount++;
      }
      if (data.parameters.width !== undefined && typeof data.parameters.width === 'number') {
        dispatch(widthChanged({ width: data.parameters.width }));
        appliedCount++;
      }
      if (data.parameters.height !== undefined && typeof data.parameters.height === 'number') {
        dispatch(heightChanged({ height: data.parameters.height }));
        appliedCount++;
      }
      if (data.parameters.seed !== undefined && typeof data.parameters.seed === 'number') {
        dispatch(setSeed(data.parameters.seed));
        appliedCount++;
      }
      if (data.parameters.steps !== undefined && typeof data.parameters.steps === 'number') {
        dispatch(setSteps(data.parameters.steps));
        appliedCount++;
      }
      if (data.parameters.cfg_scale !== undefined && typeof data.parameters.cfg_scale === 'number') {
        dispatch(setCfgScale(data.parameters.cfg_scale));
        appliedCount++;
      }

      // Handle model - requires looking up the full model config
      if (data.parameters.model !== undefined && typeof data.parameters.model === 'string') {
        dispatch(modelsApi.endpoints.getModelConfig.initiate(data.parameters.model))
          .unwrap()
          .then((modelConfig) => {
            if (modelConfig.type === 'main') {
              dispatch(modelSelected(modelConfig));
              log.debug(`Applied model: ${modelConfig.name}`);
            } else {
              log.warn(`Model ${data.parameters.model} is not a main model, skipping`);
            }
          })
          .catch((error) => {
            log.error(`Failed to load model ${data.parameters.model}: ${error}`);
          });
        appliedCount++;
      }

      if (appliedCount > 0) {
        log.info(`Applied ${appliedCount} recall parameters to store`);
      }

      // Handle LoRAs
      if (data.parameters.loras !== undefined && Array.isArray(data.parameters.loras)) {
        log.debug(`Processing ${data.parameters.loras.length} LoRA(s)`);

        // Clear existing LoRAs first
        dispatch(loraAllDeleted());

        // Add each LoRA
        for (const loraConfig of data.parameters.loras) {
          if (loraConfig.model_key && typeof loraConfig.model_key === 'string') {
            dispatch(modelsApi.endpoints.getModelConfig.initiate(loraConfig.model_key))
              .unwrap()
              .then((modelConfig) => {
                if (modelConfig.type === 'lora') {
                  const lora = {
                    id: `recalled-${Date.now()}-${Math.random()}`,
                    model: {
                      key: modelConfig.key,
                      hash: modelConfig.hash,
                      name: modelConfig.name,
                      base: modelConfig.base,
                      type: modelConfig.type,
                    },
                    weight: typeof loraConfig.weight === 'number' ? loraConfig.weight : 0.75,
                    isEnabled: typeof loraConfig.is_enabled === 'boolean' ? loraConfig.is_enabled : true,
                  };
                  dispatch(loraRecalled({ lora }));
                  log.debug(`Applied LoRA: ${modelConfig.name} (weight: ${lora.weight})`);
                } else {
                  log.warn(`Model ${loraConfig.model_key} is not a LoRA, skipping`);
                }
              })
              .catch((error) => {
                log.error(`Failed to load LoRA ${loraConfig.model_key}: ${error}`);
              });
          }
        }
        log.info(`Initiated loading of ${data.parameters.loras.length} LoRA(s)`);
      }

      // Handle Control Layers
      if (data.parameters.control_layers !== undefined && Array.isArray(data.parameters.control_layers)) {
        log.debug(`Processing ${data.parameters.control_layers.length} control layer(s)`);

        // If the list is explicitly empty, clear all existing control layers
        if (data.parameters.control_layers.length === 0) {
          dispatch(allEntitiesDeleted());
          log.info('Cleared all control layers');
        } else {
          // Replace existing control layers by first clearing them
          dispatch(allEntitiesDeleted());

          // Then add each new control layer
          data.parameters.control_layers.forEach(
            (controlConfig: {
              model_key: string;
              weight?: number;
              begin_step_percent?: number;
              end_step_percent?: number;
              control_mode?: ControlModeV2;
              image?: { image_name: string; width: number; height: number };
              processed_image?: { image_name: string; width: number; height: number };
            }) => {
              if (controlConfig.model_key && typeof controlConfig.model_key === 'string') {
                dispatch(modelsApi.endpoints.getModelConfig.initiate(controlConfig.model_key))
                  .unwrap()
                  .then(async (modelConfig) => {
                    // Pre-fetch the image DTO if an image is provided, to avoid validation errors
                    let imageObjects: Array<{
                      id: string;
                      type: 'image';
                      image: { image_name: string; width: number; height: number };
                    }> = [];
                    if (controlConfig.image?.image_name) {
                      try {
                        // Use the processed image if available, otherwise use the original
                        const imageToUse = controlConfig.processed_image || controlConfig.image;
                        await dispatch(imagesApi.endpoints.getImageDTO.initiate(imageToUse.image_name)).unwrap();
                        // Add the image to the control layer's objects array
                        imageObjects = [
                          {
                            id: `recalled-image-${Date.now()}-${Math.random()}`,
                            type: 'image' as const,
                            image: {
                              image_name: imageToUse.image_name,
                              width: imageToUse.width,
                              height: imageToUse.height,
                            },
                          },
                        ];
                        if (controlConfig.processed_image) {
                          log.debug(
                            `Pre-fetched processed control layer image: ${imageToUse.image_name} (${imageToUse.width}x${imageToUse.height})`
                          );
                        } else {
                          log.debug(
                            `Pre-fetched control layer image: ${imageToUse.image_name} (${imageToUse.width}x${imageToUse.height})`
                          );
                        }
                      } catch (imageError) {
                        log.warn(
                          `Could not pre-fetch control layer image ${controlConfig.image.image_name}, continuing without image: ${imageError}`
                        );
                      }
                    }

                    // Build a valid CanvasControlLayerState using helper function
                    const controlLayerState = getControlLayerState(`recalled-control-${Date.now()}-${Math.random()}`, {
                      objects: imageObjects,
                      controlAdapter: {
                        type: 'controlnet',
                        model: {
                          key: modelConfig.key,
                          hash: modelConfig.hash,
                          name: modelConfig.name,
                          base: modelConfig.base,
                          type: modelConfig.type,
                        },
                        weight: typeof controlConfig.weight === 'number' ? controlConfig.weight : 1.0,
                        beginEndStepPct: [
                          typeof controlConfig.begin_step_percent === 'number' ? controlConfig.begin_step_percent : 0,
                          typeof controlConfig.end_step_percent === 'number' ? controlConfig.end_step_percent : 1,
                        ] as [number, number],
                        controlMode: controlConfig.control_mode || 'balanced',
                      },
                    });

                    dispatch(controlLayerRecalled({ data: controlLayerState }));
                    log.debug(
                      `Applied control layer: ${modelConfig.name} (weight: ${controlLayerState.controlAdapter.weight})`
                    );
                    if (imageObjects.length > 0) {
                      log.info(`Control layer image loaded: ${controlConfig.image?.image_name}`);
                    }
                  })
                  .catch((error) => {
                    log.error(`Failed to load control layer ${controlConfig.model_key}: ${error}`);
                  });
              }
            }
          );
          log.info(`Initiated loading of ${data.parameters.control_layers.length} control layer(s)`);
        }
      }

      // Handle IP Adapters and model-free reference images together.
      //
      // Both ip_adapters and reference_images feed into the same refImages
      // Redux slice.  Previously they were dispatched as two independent
      // Promise.all chains — the first with replace:true, the second with
      // replace:false — which created a race: if a previous recall's
      // reference-image promises were still in-flight they could resolve
      // after the clear and re-append stale entries, doubling the list.
      //
      // Fix: collect every promise into a single array and dispatch exactly
      // once with replace:true after all of them settle.
      {
        /* eslint-disable @typescript-eslint/no-explicit-any */
        const ipAdaptersArr: any[] = Array.isArray(data.parameters.ip_adapters)
          ? (data.parameters.ip_adapters as any[])
          : [];
        const refImagesArr: any[] = Array.isArray(data.parameters.reference_images)
          ? (data.parameters.reference_images as any[])
          : [];
        /* eslint-enable @typescript-eslint/no-explicit-any */

        const hasIpAdapters = data.parameters.ip_adapters !== undefined;
        const hasRefImages = data.parameters.reference_images !== undefined;
        // Append mode (POST /api/v1/recall/{queue_id}?append=true): add the
        // recalled reference images to the existing list instead of replacing
        // it. The backend passes the flag inside the parameters dict.
        const append = data.parameters.append === true;

        if (hasIpAdapters || hasRefImages) {
          const allRefImagePromises: Promise<RefImageState | null>[] = [];

          // --- IP Adapters ---
          if (hasIpAdapters && ipAdaptersArr.length > 0) {
            log.debug(`Processing ${ipAdaptersArr.length} IP adapter(s)`);

            const ipAdapterPromises = ipAdaptersArr
              .filter((cfg: any) => cfg.model_key && typeof cfg.model_key === 'string') // eslint-disable-line @typescript-eslint/no-explicit-any
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              .map(async (adapterConfig: any): Promise<RefImageState | null> => {
                try {
                  const modelConfig = await dispatch(
                    modelsApi.endpoints.getModelConfig.initiate(adapterConfig.model_key!)
                  ).unwrap();

                  // Pre-fetch the image DTO if an image is provided, to avoid validation errors
                  if (adapterConfig.image?.image_name) {
                    try {
                      await dispatch(imagesApi.endpoints.getImageDTO.initiate(adapterConfig.image.image_name)).unwrap();
                    } catch (imageError) {
                      log.warn(
                        `Could not pre-fetch image ${adapterConfig.image.image_name}, continuing anyway: ${imageError}`
                      );
                    }
                  }

                  // Build RefImageState using helper function - supports both ip_adapter and flux_redux
                  const imageData = adapterConfig.image
                    ? {
                        original: {
                          image: {
                            image_name: adapterConfig.image.image_name,
                            width: adapterConfig.image.width ?? 512,
                            height: adapterConfig.image.height ?? 512,
                          },
                        },
                      }
                    : null;

                  const isFluxRedux = modelConfig.type === 'flux_redux';
                  const refImageState = getReferenceImageState(`recalled-ref-image-${Date.now()}-${Math.random()}`, {
                    isEnabled: true,
                    config: isFluxRedux
                      ? {
                          type: 'flux_redux',
                          image: imageData,
                          model: {
                            key: modelConfig.key,
                            hash: modelConfig.hash,
                            name: modelConfig.name,
                            base: modelConfig.base,
                            type: modelConfig.type,
                          },
                          imageInfluence: (adapterConfig.image_influence as FLUXReduxImageInfluence) || 'highest',
                        }
                      : {
                          type: 'ip_adapter',
                          image: imageData,
                          model: {
                            key: modelConfig.key,
                            hash: modelConfig.hash,
                            name: modelConfig.name,
                            base: modelConfig.base,
                            type: modelConfig.type,
                          },
                          weight: typeof adapterConfig.weight === 'number' ? adapterConfig.weight : 1.0,
                          beginEndStepPct: [
                            typeof adapterConfig.begin_step_percent === 'number' ? adapterConfig.begin_step_percent : 0,
                            typeof adapterConfig.end_step_percent === 'number' ? adapterConfig.end_step_percent : 1,
                          ] as [number, number],
                          method: (adapterConfig.method as IPMethodV2) || 'full',
                          clipVisionModel: 'ViT-H',
                        },
                  });

                  if (isFluxRedux) {
                    log.debug(`Built FLUX Redux ref image state: ${modelConfig.name}`);
                  } else {
                    log.debug(
                      `Built IP adapter ref image state: ${modelConfig.name} (weight: ${typeof adapterConfig.weight === 'number' ? adapterConfig.weight : 1.0})`
                    );
                  }
                  if (adapterConfig.image?.image_name) {
                    log.debug(
                      `IP adapter image: outputs/images/${adapterConfig.image.image_name} (${adapterConfig.image.width}x${adapterConfig.image.height})`
                    );
                  }

                  return refImageState;
                } catch (error) {
                  log.error(`Failed to load IP adapter ${adapterConfig.model_key}: ${error}`);
                  return null;
                }
              });

            allRefImagePromises.push(...ipAdapterPromises);
          }

          // --- Model-free reference images (FLUX.2 Klein, FLUX Kontext, Qwen Image Edit) ---
          // These feed the reference image directly into the main model rather than going
          // through an IP Adapter, so the backend sends them without a model_key and we
          // pick the right config type via getDefaultRefImageConfig() based on the main
          // model that is currently selected in the UI.
          if (hasRefImages && refImagesArr.length > 0) {
            log.debug(`Processing ${refImagesArr.length} reference image(s)`);

            const referenceImagePromises = refImagesArr
              .filter((cfg: any) => cfg.image?.image_name && typeof cfg.image.image_name === 'string') // eslint-disable-line @typescript-eslint/no-explicit-any
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              .map(async (refConfig: any): Promise<RefImageState | null> => {
                const imageName = refConfig.image.image_name as string;
                try {
                  // Pre-fetch the image DTO so ref image validation succeeds.
                  await dispatch(imagesApi.endpoints.getImageDTO.initiate(imageName)).unwrap();
                } catch (imageError) {
                  log.warn(`Could not pre-fetch reference image ${imageName}, continuing anyway: ${imageError}`);
                }

                // Pick the config flavor (flux2 / flux_kontext / ip_adapter fallback) that
                // matches the currently-selected main model.
                const baseConfig = getDefaultRefImageConfig(getState);
                const imageData = {
                  original: {
                    image: {
                      image_name: imageName,
                      width: typeof refConfig.image.width === 'number' ? refConfig.image.width : 512,
                      height: typeof refConfig.image.height === 'number' ? refConfig.image.height : 512,
                    },
                  },
                };

                return getReferenceImageState(`recalled-ref-image-${Date.now()}-${Math.random()}`, {
                  isEnabled: true,
                  config: { ...baseConfig, image: imageData },
                });
              });

            allRefImagePromises.push(...referenceImagePromises);
          }

          // Single dispatch after all IP adapter + reference image promises settle.
          // replace:true (the default) clears stale entries from a previous
          // recall; append mode instead pushes onto the existing list and
          // deliberately dispatches nothing when no valid states resolved, so
          // a failed append can never wipe the user's current reference images.
          Promise.all(allRefImagePromises).then((results) => {
            const validStates = results.filter((state): state is RefImageState => state !== null);
            if (append) {
              if (validStates.length > 0) {
                dispatch(refImagesRecalled({ entities: validStates, replace: false }));
                log.info(
                  `Appended ${validStates.length} reference image(s) (IP adapters + model-free) to existing list`
                );
              }
              return;
            }
            dispatch(refImagesRecalled({ entities: validStates, replace: true }));
            if (validStates.length > 0) {
              log.info(
                `Applied ${validStates.length} reference image(s) (IP adapters + model-free), replacing existing list`
              );
            } else {
              log.info('Cleared all reference images');
            }
          });
        }
      }
    }
  });

  socket.on('bulk_download_started', (data) => {
    log.debug({ data }, 'Bulk gallery download preparation started');
  });

  socket.on('bulk_download_complete', (data) => {
    log.debug({ data }, 'Bulk gallery download ready');
    const { bulk_download_item_name } = data;

    // Dismiss the "preparing" toast (which uses a prefixed id to avoid the
    // race condition where this socket event arrives before the Redux
    // middleware processes the POST response).
    toastApi.close(`preparing:${bulk_download_item_name}`);

    // The GET endpoint requires authentication, so we use fetch() with the
    // Authorization header rather than a plain <a download> link (which cannot
    // carry headers).  After fetching the blob, we create a temporary object
    // URL and trigger the browser's save dialog programmatically.
    const url = `/api/v1/images/download/${bulk_download_item_name}`;
    const token = localStorage.getItem('auth_token');
    const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};

    const handleDownload = () => {
      fetch(url, { headers })
        .then((res) => {
          if (!res.ok) {
            throw new Error(`Download failed: ${res.status}`);
          }
          return res.blob();
        })
        .then((blob) => {
          const blobUrl = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = blobUrl;
          a.download = bulk_download_item_name;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          // Delay revocation — the browser's save dialog is asynchronous,
          // and revoking immediately would invalidate the URL before the
          // download completes.
          setTimeout(() => URL.revokeObjectURL(blobUrl), 60_000);
        })
        .catch((err) => {
          log.error({ err }, 'Bulk download fetch failed');
          toast({
            id: `error:${bulk_download_item_name}`,
            title: t('gallery.bulkDownloadFailed'),
            status: 'error',
            description: String(err),
          });
        });
    };

    toast({
      id: bulk_download_item_name,
      title: t('gallery.bulkDownloadReady'),
      status: 'success',
      description: (
        <Text as="button" onClick={handleDownload} textDecoration="underline" cursor="pointer">
          {t('gallery.clickToDownload')}
        </Text>
      ),
      duration: null,
    });
  });

  socket.on('bulk_download_error', (data) => {
    log.error({ data }, 'Bulk gallery download error');

    const { bulk_download_item_name, error } = data;

    // Dismiss the "preparing" toast
    toastApi.close(`preparing:${bulk_download_item_name}`);

    toast({
      id: bulk_download_item_name,
      title: t('gallery.bulkDownloadFailed'),
      status: 'error',
      description: error,
      duration: null,
    });
  });

  socket.on('llm_task_progress', (data) => {
    log.trace({ data } as JsonObject, 'LLM task progress');
    setLLMTaskState(data.task_id, { status: 'progress', payload: data });
  });

  // Completion/error clear the entry rather than storing a terminal state. Socket
  // delivery is ordered but independent of the HTTP response, so storing a state here
  // could re-create an entry after the mutation's finally already cleared it, leaking
  // one orphan per request. The error text surfaces to the user via the RTK Query toast.
  socket.on('llm_task_complete', (data) => {
    log.trace({ data } as JsonObject, 'LLM task complete');
    clearLLMTaskState(data.task_id);
  });

  socket.on('llm_task_error', (data) => {
    log.warn({ data } as JsonObject, 'LLM task error');
    clearLLMTaskState(data.task_id);
  });
};
