import { getStore } from 'app/store/nanostores/store';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { blobToDataURL } from 'features/canvas/util/blobToDataURL';
import { isVectorMaskLayer, VECTOR_MASK_LAYER_NAME } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { renderers } from 'features/regionalPrompts/util/renderers';
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
  const state = getStore().getState();
  const { layers } = state.regionalPrompts.present;
  const { width, height } = state.regionalPrompts.present.size;
  const reduxLayers = layers.filter(isVectorMaskLayer);
  const container = document.createElement('div');
  const stage = new Konva.Stage({ container, width, height });
  renderers.renderLayers(stage, reduxLayers, 1, 'brush');

  const konvaLayers = stage.find<Konva.Layer>(`.${VECTOR_MASK_LAYER_NAME}`);
  const blobs: Record<string, Blob> = {};

  // First remove all layers
  for (const layer of konvaLayers) {
    layer.remove();
  }

  // Next render each layer to a blob
  for (const layer of konvaLayers) {
    if (layerIds && !layerIds.includes(layer.id())) {
      continue;
    }
    const reduxLayer = reduxLayers.find((l) => l.id === layer.id());
    assert(reduxLayer, `Redux layer ${layer.id()} not found`);
    stage.add(layer);
    const blob = await new Promise<Blob>((resolve) => {
      stage.toBlob({
        callback: (blob) => {
          assert(blob, 'Blob is null');
          resolve(blob);
        },
      });
    });

    if (preview) {
      const base64 = await blobToDataURL(blob);
      openBase64ImageInTab([
        {
          base64,
          caption: `${reduxLayer.id}: ${reduxLayer.positivePrompt} / ${reduxLayer.negativePrompt}`,
        },
      ]);
    }
    layer.remove();
    blobs[layer.id()] = blob;
  }

  return blobs;
};
