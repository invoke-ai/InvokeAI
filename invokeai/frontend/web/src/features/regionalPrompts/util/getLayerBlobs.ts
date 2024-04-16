import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { blobToDataURL } from 'features/canvas/util/blobToDataURL';
import { selectPromptLayerObjectGroup } from 'features/regionalPrompts/components/LayerComponent';
import { getStage, REGIONAL_PROMPT_LAYER_NAME } from 'features/regionalPrompts/store/regionalPromptsSlice';
import Konva from 'konva';
import { assert } from 'tsafe';

/**
 * Get the blobs of all regional prompt layers. Only visible layers are returned.
 * @param layerIds The IDs of the layers to get blobs for. If not provided, all regional prompt layers are used.
 * @param preview Whether to open a new tab displaying each layer.
 * @returns A map of layer IDs to blobs.
 */
export const getRegionalPromptLayerBlobs = async (
  layerIds?: string[],
  preview: boolean = false
): Promise<Record<string, Blob>> => {
  const stage = getStage();

  // This automatically omits layers that are not rendered. Rendering is controlled by the layer's `isVisible` flag in redux.
  const regionalPromptLayers = stage.getLayers().filter((l) => {
    console.log(l.name(), l.id());
    const isRegionalPromptLayer = l.name() === REGIONAL_PROMPT_LAYER_NAME;
    const isRequestedLayerId = layerIds ? layerIds.includes(l.id()) : true;
    return isRegionalPromptLayer && isRequestedLayerId;
  });

  // We need to reconstruct each layer to only output the desired data. This logic mirrors the logic in
  // `getKonvaLayerBbox()` in `invokeai/frontend/web/src/features/regionalPrompts/util/bbox.ts`
  const offscreenStageContainer = document.createElement('div');
  const offscreenStage = new Konva.Stage({
    container: offscreenStageContainer,
    width: stage.width(),
    height: stage.height(),
  });

  const blobs: Record<string, Blob> = {};

  for (const layer of regionalPromptLayers) {
    const layerClone = layer.clone();
    for (const child of layerClone.getChildren()) {
      if (selectPromptLayerObjectGroup(child)) {
        child.destroy();
      } else {
        // We need to re-cache to handle children with transparency and multiple objects - like prompt region layers.
        child.cache();
      }
    }
    offscreenStage.add(layerClone);
    const blob = await new Promise<Blob>((resolve) => {
      offscreenStage.toBlob({
        callback: (blob) => {
          assert(blob, 'Blob is null');
          resolve(blob);
        },
      });
    });

    if (preview) {
      const base64 = await blobToDataURL(blob);
      openBase64ImageInTab([{ base64, caption: layer.id() }]);
    }
    layerClone.destroy();
    blobs[layer.id()] = blob;
  }

  return blobs;
};
