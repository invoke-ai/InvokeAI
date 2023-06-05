import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { controlNetSelector } from 'features/controlNet/store/controlNetSlice';
import { nodesSelecter } from 'features/nodes/store/nodesSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { some } from 'lodash-es';

const selectIsImageInUse = createSelector(
  [
    generationSelector,
    canvasSelector,
    nodesSelecter,
    controlNetSelector,
    (state, image_name) => image_name,
  ],
  (generation, canvas, nodes, controlNet, image_name) => {
    const isInitialImage = generation.initialImage?.image_name === image_name;

    const isCanvasImage = canvas.layerState.objects.some(
      (obj) => obj.kind === 'image' && obj.image.image_name === image_name
    );

    const isNodesImage = nodes.nodes.some((node) => {
      return some(
        node.data.inputs,
        (input) =>
          input.type === 'image' && input.value?.image_name === image_name
      );
    });

    const isControlNetImage = some(
      controlNet.controlNets,
      (c) =>
        c.controlImage?.image_name === image_name ||
        c.processedControlImage?.image_name === image_name
    );

    return {
      isInitialImage,
      isCanvasImage,
      isNodesImage,
      isControlNetImage,
    };
  },
  defaultSelectorOptions
);

export const useGetIsImageInUse = (image_name?: string) => {
  const a = useAppSelector((state) => selectIsImageInUse(state, image_name));

  return a;
};
