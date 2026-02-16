import { ExternalLink, Flex, Text } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { socketConnected } from 'app/store/middleware/listenerMiddleware/listeners/socketConnected';
import type { AppStore } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { forEach, isNil, round } from 'es-toolkit/compat';
import { allEntitiesDeleted, controlLayerRecalled } from 'features/controlLayers/store/canvasSlice';
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
import { zNodeStatus } from 'features/nodes/types/invocation';
import { modelSelected } from 'features/parameters/store/actions';
import ErrorToastDescription, { getTitle } from 'features/toast/ErrorToastDescription';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { LRUCache } from 'lru-cache';
import { Trans } from 'react-i18next';
import type { ApiTagDescription } from 'services/api';
import { api, LIST_ALL_TAG, LIST_TAG } from 'services/api';
import { imagesApi } from 'services/api/endpoints/images';
import { modelsApi } from 'services/api/endpoints/models';
import { queueApi } from 'services/api/endpoints/queue';
import { buildOnInvocationComplete } from 'services/events/onInvocationComplete';
import { buildOnModelInstallError, DiscordLink, GitHubIssuesLink } from 'services/events/onModelInstallError';
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

/**
 * Sets up event listeners for the socketio client. Some components will set up their own listeners. These are the ones
 * that have app-wide implications.
 */
export const setEventListeners = ({ socket, store, setIsConnected }: SetEventListenersArg) => {
  const { dispatch, getState } = store;

  // We can have race conditions where we receive a progress event for a queue item that has already finished. Easiest
  // way to handle this is to keep track of finished queue items in a cache and ignore progress events for those.
  const finishedQueueItemIds = new LRUCache<number, boolean>({ max: 100 });

  socket.on('connect', () => {
    log.debug('Connected');
    setIsConnected(true);
    dispatch(socketConnected());
    socket.emit('subscribe_queue', { queue_id: 'default' });
    socket.emit('subscribe_bulk_download', { bulk_download_id: 'default' });
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
    if (finishedQueueItemIds.has(data.item_id)) {
      log.trace({ data }, `Received event for already-finished queue item ${data.item_id}`);
      return;
    }

    // we've got new status for the queue item, batch and queue
    const {
      item_id,
      status,
      batch_status,
      error_type,
      error_message,
      destination,
      started_at,
      updated_at,
      completed_at,
      error_traceback,
    } = data;

    log.debug({ data }, `Queue item ${item_id} status updated: ${status}`);

    // // Update this specific queue item in the list of queue items
    dispatch(
      queueApi.util.updateQueryData('getQueueItem', item_id, (draft) => {
        draft.status = status;
        draft.started_at = started_at;
        draft.updated_at = updated_at;
        draft.completed_at = completed_at;
        draft.error_type = error_type;
        draft.error_message = error_message;
        draft.error_traceback = error_traceback;
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
        toast({
          id: `INVOCATION_ERROR_${error_type}`,
          title: getTitle(error_type),
          status: 'error',
          duration: null,
          updateDescription: true,
          description: <ErrorToastDescription errorType={error_type} errorMessage={error_message} />,
        });
      }
      // If the queue item is completed, failed, or cancelled, we want to clear the last progress event
      $lastProgressEvent.set(null);
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

      // Handle IP Adapters as Reference Images
      if (data.parameters.ip_adapters !== undefined && Array.isArray(data.parameters.ip_adapters)) {
        log.debug(`Processing ${data.parameters.ip_adapters.length} IP adapter(s)`);

        // If the list is explicitly empty, clear existing reference images
        if (data.parameters.ip_adapters.length === 0) {
          dispatch(refImagesRecalled({ entities: [], replace: true }));
          log.info('Cleared all IP adapter reference images');
        } else {
          // Build promises for all IP adapters, then dispatch once with replace: true
          const ipAdapterPromises = data.parameters.ip_adapters
            .filter((cfg) => cfg.model_key && typeof cfg.model_key === 'string')
            .map(async (adapterConfig) => {
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

          // Wait for all IP adapters to load, then dispatch with replace: true
          Promise.all(ipAdapterPromises).then((refImageStates) => {
            const validStates = refImageStates.filter((state): state is RefImageState => state !== null);
            if (validStates.length > 0) {
              dispatch(refImagesRecalled({ entities: validStates, replace: true }));
              log.info(`Applied ${validStates.length} IP adapter(s), replacing existing list`);
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
