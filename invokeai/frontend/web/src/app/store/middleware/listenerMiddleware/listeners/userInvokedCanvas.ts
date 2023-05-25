import { startAppListening } from '..';
import { nodeUpdated, sessionCreated } from 'services/thunks/session';
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
 * This listener is responsible invoking the canvas. This involved a number of steps:
 *
 * 1. Generate image blobs from the canvas layers
 * 2. Determine the generation mode from the layers (txt2img, img2img, inpaint)
 * 3. Build the canvas graph
 * 4. Create the session
 * 5. Upload the init image if necessary, then update the graph to refer to it (needs a separate request)
 * 6. Upload the mask image if necessary, then update the graph to refer to it (needs a separate request)
 * 7. Initialize the staging area if not yet initialized
 * 8. Finally, dispatch the sessionReadyToInvoke action to invoke the session
 *
 * We have to do the uploads after creating the session:
 * - We need to associate these particular uploads to a session, and flag them as intermediates
 * - To do this, we need to associa
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

      moduleLog({ data: graph }, 'Canvas graph built');

      // If we are generating img2img or inpaint, we need to upload the init images
      if (baseNode.type === 'img2img' || baseNode.type === 'inpaint') {
        const baseFilename = `${uuidv4()}.png`;
        dispatch(
          imageUploaded({
            formData: {
              file: new File([baseBlob], baseFilename, { type: 'image/png' }),
            },
            isIntermediate: true,
          })
        );

        // Wait for the image to be uploaded
        const [{ payload: basePayload }] = await take(
          (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
            imageUploaded.fulfilled.match(action) &&
            action.meta.arg.formData.file.name === baseFilename
        );

        // Update the base node with the image name and type
        const { image_name: baseName, image_type: baseType } =
          basePayload.response;

        baseNode.image = {
          image_name: baseName,
          image_type: baseType,
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
            isIntermediate: true,
          })
        );

        // Wait for the mask to be uploaded
        const [{ payload: maskPayload }] = await take(
          (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
            imageUploaded.fulfilled.match(action) &&
            action.meta.arg.formData.file.name === maskFilename
        );

        // Update the base node with the image name and type
        const { image_name: maskName, image_type: maskType } =
          maskPayload.response;

        baseNode.mask = {
          image_name: maskName,
          image_type: maskType,
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
            imageType: baseNode.image.image_type,
            requestBody: { session_id: sessionId },
          })
        );
      }

      // Associate the mask image with the session, now that we have the session ID
      if (baseNode.type === 'inpaint' && baseNode.mask) {
        dispatch(
          imageUpdated({
            imageName: baseNode.mask.image_name,
            imageType: baseNode.mask.image_type,
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
