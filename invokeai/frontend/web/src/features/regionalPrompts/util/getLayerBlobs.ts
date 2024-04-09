import { getStore } from 'app/store/nanostores/store';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { blobToDataURL } from 'features/canvas/util/blobToDataURL';
import { $stage } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { assert } from 'tsafe';

export const getLayerBlobs = async () => {
  const state = getStore().getState();
  const stage = $stage.get();
  assert(stage !== null, 'Stage is null');
  const stageLayers = stage.getLayers().filter((l) => l.name() === 'regionalPromptLayer');
  for (const layer of stageLayers) {
    const blob = await new Promise<Blob>((resolve) => {
      layer.toBlob({
        callback: (blob) => {
          assert(blob, 'Blob is null');
          resolve(blob);
        },
      });
    });
    const base64 = await blobToDataURL(blob);
    const prompt = state.regionalPrompts.layers.find((l) => l.id === layer.id())?.prompt;
    assert(prompt !== undefined, 'Prompt is undefined');
    openBase64ImageInTab([{ base64, caption: prompt }]);
  }
};
