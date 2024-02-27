import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { parseify } from 'common/util/serialize';
import { canvasBatchIdAdded, stagingAreaInitialized } from 'features/canvas/store/canvasSlice';
import { blobToDataURL } from 'features/canvas/util/blobToDataURL';
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { getCanvasGenerationMode } from 'features/canvas/util/getCanvasGenerationMode';
import { canvasGraphBuilt } from 'features/nodes/store/actions';
import { buildCanvasGraph } from 'features/nodes/util/graph/buildCanvasGraph';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { imagesApi } from 'services/api/endpoints/images';
import { queueApi } from 'services/api/endpoints/queue';
import type { ImageDTO } from 'services/api/types';

/**
 * This listener is responsible invoking the canvas. This involves a number of steps:
 *
 * 1. Generate image blobs from the canvas layers
 * 2. Determine the generation mode from the layers (txt2img, img2img, inpaint)
 * 3. Build the canvas graph
 * 4. Create the session with the graph
 * 5. Upload the init image if necessary
 * 6. Upload the mask image if necessary
 * 7. Update the init and mask images with the session ID
 * 8. Initialize the staging area if not yet initialized
 * 9. Dispatch the sessionReadyToInvoke action to invoke the session
 */
export const addEnqueueRequestedCanvasListener = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'unifiedCanvas',
    effect: async (action, { getState, dispatch }) => {
      const log = logger('queue');
      const { prepend } = action.payload;
      const state = getState();

      const { layerState, boundingBoxCoordinates, boundingBoxDimensions, isMaskEnabled, shouldPreserveMaskedArea } =
        state.canvas;

      // Build canvas blobs
      const canvasBlobsAndImageData = await getCanvasData(
        layerState,
        boundingBoxCoordinates,
        boundingBoxDimensions,
        isMaskEnabled,
        shouldPreserveMaskedArea
      );

      if (!canvasBlobsAndImageData) {
        log.error('Unable to create canvas data');
        return;
      }

      const { baseBlob, baseImageData, maskBlob, maskImageData } = canvasBlobsAndImageData;

      // Determine the generation mode
      const generationMode = getCanvasGenerationMode(baseImageData, maskImageData);

      if (state.system.enableImageDebugging) {
        const baseDataURL = await blobToDataURL(baseBlob);
        const maskDataURL = await blobToDataURL(maskBlob);
        openBase64ImageInTab([
          { base64: maskDataURL, caption: 'mask b64' },
          { base64: baseDataURL, caption: 'image b64' },
        ]);
      }

      log.debug(`Generation mode: ${generationMode}`);

      // Temp placeholders for the init and mask images
      let canvasInitImage: ImageDTO | undefined;
      let canvasMaskImage: ImageDTO | undefined;

      // For img2img and inpaint/outpaint, we need to upload the init images
      if (['img2img', 'inpaint', 'outpaint'].includes(generationMode)) {
        // upload the image, saving the request id
        canvasInitImage = await dispatch(
          imagesApi.endpoints.uploadImage.initiate({
            file: new File([baseBlob], 'canvasInitImage.png', {
              type: 'image/png',
            }),
            image_category: 'general',
            is_intermediate: true,
          })
        ).unwrap();
      }

      // For inpaint/outpaint, we also need to upload the mask layer
      if (['inpaint', 'outpaint'].includes(generationMode)) {
        // upload the image, saving the request id
        canvasMaskImage = await dispatch(
          imagesApi.endpoints.uploadImage.initiate({
            file: new File([maskBlob], 'canvasMaskImage.png', {
              type: 'image/png',
            }),
            image_category: 'mask',
            is_intermediate: true,
          })
        ).unwrap();
      }

      const graph = buildCanvasGraph(state, generationMode, canvasInitImage, canvasMaskImage);

      log.debug({ graph: parseify(graph) }, `Canvas graph built`);

      // currently this action is just listened to for logging
      dispatch(canvasGraphBuilt(graph));

      const batchConfig = prepareLinearUIBatch(state, graph, prepend);

      try {
        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
            fixedCacheKey: 'enqueueBatch',
          })
        );

        const enqueueResult = await req.unwrap();
        req.reset();

        const batchId = enqueueResult.batch.batch_id as string; // we know the is a string, backend provides it

        // Prep the canvas staging area if it is not yet initialized
        if (!state.canvas.layerState.stagingArea.boundingBox) {
          dispatch(
            stagingAreaInitialized({
              boundingBox: {
                ...state.canvas.boundingBoxCoordinates,
                ...state.canvas.boundingBoxDimensions,
              },
            })
          );
        }

        // Associate the session with the canvas session ID
        dispatch(canvasBatchIdAdded(batchId));
      } catch {
        // no-op
      }
    },
  });
};
