import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasState } from 'features/controlLayers/store/types';
import { selectDeleteImageModalSlice } from 'features/deleteImageModal/store/slice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState } from 'features/nodes/store/types';
import { isImageFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { UpscaleState } from 'features/parameters/store/upscaleSlice';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { some } from 'lodash-es';

import type { ImageUsage } from './types';
// TODO(psyche): handle image deletion (canvas staging area?)
export const getImageUsage = (nodes: NodesState, canvas: CanvasState, upscale: UpscaleState, image_name: string) => {
  const isNodesImage = nodes.nodes
    .filter(isInvocationNode)
    .some((node) =>
      some(node.data.inputs, (input) => isImageFieldInputInstance(input) && input.value?.image_name === image_name)
    );

  const isUpscaleImage = upscale.upscaleInitialImage?.image_name === image_name;

  const isReferenceImage = canvas.referenceImages.entities.some(
    ({ ipAdapter }) => ipAdapter.image?.image_name === image_name
  );

  const isRasterLayerImage = canvas.rasterLayers.entities.some(({ objects }) =>
    objects.some((obj) => obj.type === 'image' && obj.image.image_name === image_name)
  );

  const isControlLayerImage = canvas.controlLayers.entities.some(({ objects }) =>
    objects.some((obj) => obj.type === 'image' && obj.image.image_name === image_name)
  );

  const isInpaintMaskImage = canvas.inpaintMasks.entities.some(({ objects }) =>
    objects.some((obj) => obj.type === 'image' && obj.image.image_name === image_name)
  );

  const isRegionalGuidanceImage = canvas.regionalGuidance.entities.some(({ referenceImages }) =>
    referenceImages.some(({ ipAdapter }) => ipAdapter.image?.image_name === image_name)
  );

  const imageUsage: ImageUsage = {
    isUpscaleImage,
    isRasterLayerImage,
    isInpaintMaskImage,
    isRegionalGuidanceImage,
    isNodesImage,
    isControlLayerImage,
    isReferenceImage,
  };

  return imageUsage;
};

export const selectImageUsage = createMemoizedSelector(
  selectDeleteImageModalSlice,
  selectNodesSlice,
  selectCanvasSlice,
  selectUpscaleSlice,
  (deleteImageModal, nodes, canvas, upscale) => {
    const { imagesToDelete } = deleteImageModal;

    if (!imagesToDelete.length) {
      return [];
    }

    const imagesUsage = imagesToDelete.map((i) => getImageUsage(nodes, canvas, upscale, i.image_name));

    return imagesUsage;
  }
);
