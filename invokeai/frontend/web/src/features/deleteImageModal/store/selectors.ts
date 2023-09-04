import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { some } from 'lodash-es';
import { ImageUsage } from './types';
import { isInvocationNode } from 'features/nodes/types/types';

export const getImageUsage = (state: RootState, image_name: string) => {
  const { generation, canvas, nodes, controlNet } = state;
  const isInitialImage = generation.initialImage?.imageName === image_name;

  const isCanvasImage = canvas.layerState.objects.some(
    (obj) => obj.kind === 'image' && obj.imageName === image_name
  );

  const isNodesImage = nodes.nodes.filter(isInvocationNode).some((node) => {
    return some(
      node.data.inputs,
      (input) =>
        input.type === 'ImageField' && input.value?.image_name === image_name
    );
  });

  const isControlNetImage = some(
    controlNet.controlNets,
    (c) =>
      c.controlImage === image_name || c.processedControlImage === image_name
  );

  const imageUsage: ImageUsage = {
    isInitialImage,
    isCanvasImage,
    isNodesImage,
    isControlNetImage,
  };

  return imageUsage;
};

export const selectImageUsage = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { imagesToDelete } = state.deleteImageModal;

    if (!imagesToDelete.length) {
      return [];
    }

    const imagesUsage = imagesToDelete.map((i) =>
      getImageUsage(state, i.image_name)
    );

    return imagesUsage;
  },
  defaultSelectorOptions
);
