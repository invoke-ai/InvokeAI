import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { buildPromptExpansionGraph } from 'features/nodes/util/graph/buildPromptExpansionGraph';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { uploadImages } from 'services/api/endpoints/images';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { EnqueueBatchArg, ImageDTO } from 'services/api/types';
import { $promptExpansionRequest } from 'services/events/stores';
import type { JsonObject } from 'type-fest';

const log = logger('queue');

export const promptExpansionRequested = createAction(`prompt/expansionRequested`);
export const promptGenerationFromImageRequested = createAction<{ imageDTO: ImageDTO }>(
  `prompt/generationFromImageRequested`
);
export const promptGenerationFromUploadRequested = createAction<{ file: File }>(`prompt/generationFromUploadRequested`);

export const addPromptExpansionRequestedListener = (startAppListening: AppStartListening) => {
  // Handle prompt expansion (from button click)
  startAppListening({
    actionCreator: promptExpansionRequested,
    effect: async (action, { dispatch, getState }) => {
      const state = getState();

      const enqueueBatchArg: EnqueueBatchArg = {
        prepend: true,
        batch: {
          graph: buildPromptExpansionGraph({
            state,
          }),
          runs: 1,
        },
      };

      try {
        // Track the prompt expansion request
        $promptExpansionRequest.set({
          startTime: Date.now(),
          status: 'pending' as const,
        });

        log.debug('Prompt expansion request initiated');

        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(enqueueBatchArg, enqueueMutationFixedCacheKeyOptions)
        );

        const enqueueResult = await req.unwrap();

        req.reset();
        log.debug({ enqueueResult } as JsonObject, t('queue.graphQueued'));
      } catch (error) {
        log.error({ enqueueBatchArg } as JsonObject, t('queue.graphFailedToQueue'));

        if (error instanceof Object && 'status' in error && error.status === 403) {
          return;
        } else {
          toast({
            id: 'GRAPH_QUEUE_FAILED',
            title: t('queue.graphFailedToQueue'),
            status: 'error',
          });
        }
      }
    },
  });

  // Handle prompt generation from image (from drag & drop or context menu)
  startAppListening({
    actionCreator: promptGenerationFromImageRequested,
    effect: async (action, { dispatch, getState }) => {
      const { imageDTO } = action.payload;
      const state = getState();

      const enqueueBatchArg: EnqueueBatchArg = {
        prepend: true,
        batch: {
          graph: buildPromptExpansionGraph({
            state,
            imageDTO,
          }),
          runs: 1,
        },
      };

      try {
        // Track the prompt generation request
        $promptExpansionRequest.set({
          startTime: Date.now(),
          status: 'pending' as const,
        });

        log.debug({ imageDTO: imageDTO.image_name }, 'Prompt generation from image request initiated');

        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(enqueueBatchArg, enqueueMutationFixedCacheKeyOptions)
        );

        const enqueueResult = await req.unwrap();

        req.reset();
        log.debug({ enqueueResult } as JsonObject, t('queue.graphQueued'));
      } catch (error) {
        log.error({ enqueueBatchArg } as JsonObject, t('queue.graphFailedToQueue'));

        if (error instanceof Object && 'status' in error && error.status === 403) {
          return;
        } else {
          toast({
            id: 'GRAPH_QUEUE_FAILED',
            title: t('queue.graphFailedToQueue'),
            status: 'error',
          });
        }
      }
    },
  });

  // Handle prompt generation from uploaded image
  startAppListening({
    actionCreator: promptGenerationFromUploadRequested,
    effect: async (action, { dispatch, getState }) => {
      const { file } = action.payload;
      const state = getState();

      try {
        // Upload the image first
        log.debug({ fileName: file.name }, 'Uploading image for prompt generation');

        const imageDTOs = await uploadImages([
          {
            file,
            image_category: 'user',
            is_intermediate: false,
            silent: true,
          },
        ]);

        if (imageDTOs.length === 0) {
          throw new Error('Failed to upload image');
        }

        const imageDTO = imageDTOs[0];

        // Now generate prompt from the uploaded image
        const enqueueBatchArg: EnqueueBatchArg = {
          prepend: true,
          batch: {
            graph: buildPromptExpansionGraph({
              state,
              imageDTO,
            }),
            runs: 1,
          },
        };

        // Track the prompt generation request
        $promptExpansionRequest.set({
          startTime: Date.now(),
          status: 'pending' as const,
        });

        log.debug({ imageDTO: imageDTO.image_name }, 'Prompt generation from uploaded image request initiated');

        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(enqueueBatchArg, enqueueMutationFixedCacheKeyOptions)
        );

        const enqueueResult = await req.unwrap();

        req.reset();
        log.debug({ enqueueResult } as JsonObject, t('queue.graphQueued'));
      } catch (error) {
        log.error({ error } as JsonObject, 'Failed to upload image and generate prompt');

        if (error instanceof Object && 'status' in error && error.status === 403) {
          return;
        } else {
          toast({
            id: 'UPLOAD_AND_PROMPT_GENERATION_FAILED',
            title: t('toast.uploadAndPromptGenerationFailed'),
            status: 'error',
          });
        }
      }
    },
  });
};
