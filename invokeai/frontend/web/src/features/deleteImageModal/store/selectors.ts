import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import type { CanvasState } from 'features/canvas/store/canvasTypes';
import {
  selectControlAdapterAll,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdaptersState } from 'features/controlAdapters/store/types';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { selectDeleteImageModalSlice } from 'features/deleteImageModal/store/slice';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { NodesState } from 'features/nodes/store/types';
import { isImageFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import type { GenerationState } from 'features/parameters/store/types';
import { some } from 'lodash-es';

import type { ImageUsage } from './types';

export const getImageUsage = (
  generation: GenerationState,
  canvas: CanvasState,
  nodes: NodesState,
  controlAdapters: ControlAdaptersState,
  image_name: string
) => {
  const isInitialImage = generation.initialImage?.imageName === image_name;

  const isCanvasImage = canvas.layerState.objects.some((obj) => obj.kind === 'image' && obj.imageName === image_name);

  const isNodesImage = nodes.nodes.filter(isInvocationNode).some((node) => {
    return some(
      node.data.inputs,
      (input) => isImageFieldInputInstance(input) && input.value?.image_name === image_name
    );
  });

  const isControlImage = selectControlAdapterAll(controlAdapters).some(
    (ca) => ca.controlImage === image_name || (isControlNetOrT2IAdapter(ca) && ca.processedControlImage === image_name)
  );

  const imageUsage: ImageUsage = {
    isInitialImage,
    isCanvasImage,
    isNodesImage,
    isControlImage,
  };

  return imageUsage;
};

export const selectImageUsage = createMemoizedSelector(
  selectDeleteImageModalSlice,
  selectGenerationSlice,
  selectCanvasSlice,
  selectNodesSlice,
  selectControlAdaptersSlice,
  (deleteImageModal, generation, canvas, nodes, controlAdapters) => {
    const { imagesToDelete } = deleteImageModal;

    if (!imagesToDelete.length) {
      return [];
    }

    const imagesUsage = imagesToDelete.map((i) =>
      getImageUsage(generation, canvas, nodes, controlAdapters, i.image_name)
    );

    return imagesUsage;
  }
);
