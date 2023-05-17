import { startAppListening } from '..';
import { sessionCreated, sessionInvoked } from 'services/thunks/session';
import { buildCanvasGraphComponents } from 'features/nodes/util/graphBuilders/buildCanvasGraph';
import { log } from 'app/logging/useLogger';
import { canvasGraphBuilt } from 'features/nodes/store/actions';
import { imageUploaded } from 'services/thunks/image';
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

const moduleLog = log.child({ namespace: 'invoke' });

/**
 * This listener is responsible for building the canvas graph and blobs when the user invokes the canvas.
 * It is also responsible for uploading the base and mask layers to the server.
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

      // Upload the base layer, to be used as init image
      const baseFilename = `${uuidv4()}.png`;

      dispatch(
        imageUploaded({
          imageType: 'intermediates',
          formData: {
            file: new File([baseBlob], baseFilename, { type: 'image/png' }),
          },
        })
      );

      if (baseNode.type === 'img2img' || baseNode.type === 'inpaint') {
        const [{ payload: basePayload }] = await take(
          (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
            imageUploaded.fulfilled.match(action) &&
            action.meta.arg.formData.file.name === baseFilename
        );

        const { image_name: baseName, image_type: baseType } =
          basePayload.response;

        baseNode.image = {
          image_name: baseName,
          image_type: baseType,
        };
      }

      // Upload the mask layer image
      const maskFilename = `${uuidv4()}.png`;

      if (baseNode.type === 'inpaint') {
        dispatch(
          imageUploaded({
            imageType: 'intermediates',
            formData: {
              file: new File([maskBlob], maskFilename, { type: 'image/png' }),
            },
          })
        );

        const [{ payload: maskPayload }] = await take(
          (action): action is ReturnType<typeof imageUploaded.fulfilled> =>
            imageUploaded.fulfilled.match(action) &&
            action.meta.arg.formData.file.name === maskFilename
        );

        const { image_name: maskName, image_type: maskType } =
          maskPayload.response;

        baseNode.mask = {
          image_name: maskName,
          image_type: maskType,
        };
      }

      // Assemble!
      const nodes: Graph['nodes'] = {
        [rangeNode.id]: rangeNode,
        [iterateNode.id]: iterateNode,
        [baseNode.id]: baseNode,
      };

      const graph = { nodes, edges };

      dispatch(canvasGraphBuilt(graph));
      moduleLog({ data: graph }, 'Canvas graph built');

      // Actually create the session
      dispatch(sessionCreated({ graph }));

      // Wait for the session to be invoked (this is just the HTTP request to start processing)
      const [{ meta }] = await take(sessionInvoked.fulfilled.match);

      const { sessionId } = meta.arg;

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

      dispatch(canvasSessionIdChanged(sessionId));
    },
  });
};
