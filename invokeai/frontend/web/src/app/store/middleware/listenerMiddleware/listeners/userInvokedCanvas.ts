import { log } from 'app/logging/useLogger';
import { userInvoked } from 'app/store/actions';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import {
  canvasSessionIdChanged,
  stagingAreaInitialized,
} from 'features/canvas/store/canvasSlice';
import { blobToDataURL } from 'features/canvas/util/blobToDataURL';
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { getCanvasGenerationMode } from 'features/canvas/util/getCanvasGenerationMode';
import { canvasGraphBuilt } from 'features/nodes/store/actions';
import { buildCanvasGraph } from 'features/nodes/util/graphBuilders/buildCanvasGraph';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { imagesApi } from 'services/api/endpoints/images';
import { sessionCreated } from 'services/api/thunks/session';
import { ImageDTO } from 'services/api/types';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'invoke' });

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
export const addUserInvokedCanvasListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof userInvoked> =>
      userInvoked.match(action) && action.payload === 'unifiedCanvas',
    effect: async (action, { getState, dispatch, take }) => {
      const state = getState();

      // Build canvas blobs
      const canvasBlobsAndImageData = await getCanvasData(state);

      if (!canvasBlobsAndImageData) {
        moduleLog.error('Unable to create canvas data');
        return;
      }

      const { baseBlob, baseImageData, maskBlob, maskImageData } =
        canvasBlobsAndImageData;

      // Determine the generation mode
      const generationMode = getCanvasGenerationMode(
        baseImageData,
        maskImageData
      );

      if (state.system.enableImageDebugging) {
        const baseDataURL = await blobToDataURL(baseBlob);
        const maskDataURL = await blobToDataURL(maskBlob);
        openBase64ImageInTab([
          { base64: maskDataURL, caption: 'mask b64' },
          { base64: baseDataURL, caption: 'image b64' },
        ]);
      }

      moduleLog.debug(`Generation mode: ${generationMode}`);

      // Temp placeholders for the init and mask images
      let canvasInitImage: ImageDTO | undefined;
      let canvasMaskImage: ImageDTO | undefined;

      // For img2img and inpaint/outpaint, we need to upload the init images
      if (['img2img', 'inpaint', 'outpaint'].includes(generationMode)) {
        // upload the image, saving the request id
        const { requestId: initImageUploadedRequestId } = dispatch(
          imagesApi.endpoints.uploadImage.initiate({
            file: new File([baseBlob], 'canvasInitImage.png', {
              type: 'image/png',
            }),
            image_category: 'general',
            is_intermediate: true,
          })
        );

        // Wait for the image to be uploaded, matching by request id
        const [{ payload }] = await take(
          // TODO: figure out how to narrow this action's type
          (action) =>
            imagesApi.endpoints.uploadImage.matchFulfilled(action) &&
            action.meta.requestId === initImageUploadedRequestId
        );

        canvasInitImage = payload as ImageDTO;
      }

      // For inpaint/outpaint, we also need to upload the mask layer
      if (['inpaint', 'outpaint'].includes(generationMode)) {
        // upload the image, saving the request id
        const { requestId: maskImageUploadedRequestId } = dispatch(
          imagesApi.endpoints.uploadImage.initiate({
            file: new File([maskBlob], 'canvasMaskImage.png', {
              type: 'image/png',
            }),
            image_category: 'mask',
            is_intermediate: true,
          })
        );

        // Wait for the image to be uploaded, matching by request id
        const [{ payload }] = await take(
          // TODO: figure out how to narrow this action's type
          (action) =>
            imagesApi.endpoints.uploadImage.matchFulfilled(action) &&
            action.meta.requestId === maskImageUploadedRequestId
        );

        canvasMaskImage = payload as ImageDTO;
      }

      const graph = buildCanvasGraph(
        state,
        generationMode,
        canvasInitImage,
        canvasMaskImage
      );

      moduleLog.debug({ graph }, `Canvas graph built`);

      // currently this action is just listened to for logging
      dispatch(canvasGraphBuilt(graph));

      // Create the session, store the request id
      const { requestId: sessionCreatedRequestId } = dispatch(
        sessionCreated({ graph })
      );

      // Take the session created action, matching by its request id
      const [sessionCreatedAction] = await take(
        (action): action is ReturnType<typeof sessionCreated.fulfilled> =>
          sessionCreated.fulfilled.match(action) &&
          action.meta.requestId === sessionCreatedRequestId
      );
      const sessionId = sessionCreatedAction.payload.id;

      // Associate the init image with the session, now that we have the session ID
      if (['img2img', 'inpaint'].includes(generationMode) && canvasInitImage) {
        dispatch(
          imagesApi.endpoints.updateImage.initiate({
            imageDTO: canvasInitImage,
            changes: { session_id: sessionId },
          })
        );
      }

      // Associate the mask image with the session, now that we have the session ID
      if (['inpaint'].includes(generationMode) && canvasMaskImage) {
        dispatch(
          imagesApi.endpoints.updateImage.initiate({
            imageDTO: canvasMaskImage,
            changes: { session_id: sessionId },
          })
        );
      }

      // Prep the canvas staging area if it is not yet initialized
      if (!state.canvas.layerState.stagingArea.boundingBox) {
        dispatch(
          stagingAreaInitialized({
            sessionId,
            boundingBox: {
              ...state.canvas.boundingBoxCoordinates,
              ...state.canvas.boundingBoxDimensions,
            },
          })
        );
      }

      // Flag the session with the canvas session ID
      dispatch(canvasSessionIdChanged(sessionId));

      // We are ready to invoke the session!
      dispatch(sessionReadyToInvoke());
    },
  });
};
