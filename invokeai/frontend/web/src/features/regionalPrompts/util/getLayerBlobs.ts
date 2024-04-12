import { getStore } from 'app/store/nanostores/store';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { blobToDataURL } from 'features/canvas/util/blobToDataURL';
import { selectPromptLayerObjectGroup } from 'features/regionalPrompts/components/LayerComponent';
import { $stage, REGIONAL_PROMPT_LAYER_NAME } from 'features/regionalPrompts/store/regionalPromptsSlice';
import Konva from 'konva';
import { assert } from 'tsafe';

export const getRegionalPromptLayerBlobs = async (preview: boolean = false): Promise<Record<string, Blob>> => {
  const state = getStore().getState();
  const stage = $stage.get();
  assert(stage !== null, 'Stage is null');
  const regionalPromptLayers = stage.getLayers().filter((l) => l.name() === REGIONAL_PROMPT_LAYER_NAME);

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
    for (const child of layerClone.getChildren(selectPromptLayerObjectGroup)) {
      child.destroy();
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
    blobs[layer.id()] = blob;

    if (preview) {
      const base64 = await blobToDataURL(blob);
      const prompt = state.regionalPrompts.layers.find((l) => l.id === layer.id())?.prompt;
      openBase64ImageInTab([{ base64, caption: prompt ?? '' }]);
    }
    layerClone.destroy();
  }

  return blobs;
};
