import { startAppListening } from '..';
import { sessionCreated } from 'services/thunks/session';
import { buildCanvasGraphComponents } from 'features/nodes/util/graphBuilders/buildCanvasGraph';
import { log } from 'app/logging/useLogger';
import { canvasGraphBuilt } from 'features/nodes/store/actions';
import { imageUpdated, imageUploaded } from 'services/thunks/image';
import { v4 as uuidv4 } from 'uuid';
import { Graph } from 'services/api';
import {
  canvasSessionIdChanged,
  stagingAreaInitialized,
} from 'features/canvas/store/canvasSlice';
import { userInvoked } from 'app/store/actions';
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { getCanvasGenerationMode } from 'features/canvas/util/getCanvasGenerationMode';
import { blobToDataURL } from 'features/canvas/util/blobToDataURL';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { sessionReadyToInvoke } from 'features/system/store/actions';

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

      // Build the canvas graph
      const graphComponents = await buildCanvasGraphComponents(
        state,
        generationMode
      );

      if (!graphComponents) {
        moduleLog.error('Problem building graph');
        return;
      }

      const { rangeNode, iterateNode, baseNode, edges } = graphComponents;

      // Assemble! Note that this graph *does not have the init or mask image set yet!*
      const nodes: Graph['nodes'] = {
        [rangeNode.id]: rangeNode,
        [iterateNode.id]: iterateNode,
        [baseNode.id]: baseNode,
      };

      const graph = { nodes, edges };

      dispatch(canvasGraphBuilt(graph));

      moduleLog.debug({ data: graph }, 'Canvas graph built');

      // If we are generating img2img or inpaint, we need to upload the init images
      if (baseNode.type === 'img2img' || baseNode.type === 'inpaint') {
        const baseFilename = `${uuidv4()}.png`;
        dispatch(
          imageUploaded({
            formData: {
              file: new File([baseBlob], baseFilename, { type: 'image/png' }),
            },
            imageCategory: 'general',
            isIntermediate: true,
          })
        );

        // Wait for the image to be uploaded
        const [{ payload: baseImageDTO }] = await take(
          (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
            imageUploaded.fulfilled.match(action) &&
            action.meta.arg.formData.file.name === baseFilename
        );

        // Update the base node with the image name and type
        baseNode.image = {
          image_name: baseImageDTO.image_name,
          image_origin: baseImageDTO.image_origin,
        };
      }

      // For inpaint, we also need to upload the mask layer
      if (baseNode.type === 'inpaint') {
        const maskFilename = `${uuidv4()}.png`;
        dispatch(
          imageUploaded({
            formData: {
              file: new File([maskBlob], maskFilename, { type: 'image/png' }),
            },
            imageCategory: 'mask',
            isIntermediate: true,
          })
        );

        // Wait for the mask to be uploaded
        const [{ payload: maskImageDTO }] = await take(
          (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
            imageUploaded.fulfilled.match(action) &&
            action.meta.arg.formData.file.name === maskFilename
        );

        // Update the base node with the image name and type
        baseNode.mask = {
          image_name: maskImageDTO.image_name,
          image_origin: maskImageDTO.image_origin,
        };
      }

      // Create the session and wait for response
      dispatch(sessionCreated({ graph }));
      const [sessionCreatedAction] = await take(sessionCreated.fulfilled.match);
      const sessionId = sessionCreatedAction.payload.id;

      // Associate the init image with the session, now that we have the session ID
      if (
        (baseNode.type === 'img2img' || baseNode.type === 'inpaint') &&
        baseNode.image
      ) {
        dispatch(
          imageUpdated({
            imageName: baseNode.image.image_name,
            imageOrigin: baseNode.image.image_origin,
            requestBody: { session_id: sessionId },
          })
        );
      }

      // Associate the mask image with the session, now that we have the session ID
      if (baseNode.type === 'inpaint' && baseNode.mask) {
        dispatch(
          imageUpdated({
            imageName: baseNode.mask.image_name,
            imageOrigin: baseNode.mask.image_origin,
            requestBody: { session_id: sessionId },
          })
        );
      }

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
