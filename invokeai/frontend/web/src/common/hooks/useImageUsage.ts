import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { controlNetSelector } from 'features/controlNet/store/controlNetSlice';
import { nodesSelecter } from 'features/nodes/store/nodesSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { some } from 'lodash-es';

export type ImageUsage = {
  isInitialImage: boolean;
  isCanvasImage: boolean;
  isNodesImage: boolean;
  isControlNetImage: boolean;
};

const selectImageUsage = createSelector(
  [
    generationSelector,
    canvasSelector,
    nodesSelecter,
    controlNetSelector,
    (state: RootState, image_name?: string) => image_name,
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

    const imageUsage: ImageUsage = {
      isInitialImage,
      isCanvasImage,
      isNodesImage,
      isControlNetImage,
    };

    return imageUsage;
  },
  defaultSelectorOptions
);

export const useImageUsage = (image_name?: string) => {
  const imageUsage = useAppSelector((state) =>
    selectImageUsage(state, image_name)
  );

  return imageUsage;
};
