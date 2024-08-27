import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasState } from 'features/controlLayers/store/types';
import { selectDeleteImageModalSlice } from 'features/deleteImageModal/store/slice';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { NodesState } from 'features/nodes/store/types';
import { isImageFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { some } from 'lodash-es';

import type { ImageUsage } from './types';
// TODO(psyche): handle image deletion (canvas sessions?)
export const getImageUsage = (nodes: NodesState, canvas: CanvasState, image_name: string) => {
  const isNodesImage = nodes.nodes
    .filter(isInvocationNode)
    .some((node) =>
      some(node.data.inputs, (input) => isImageFieldInputInstance(input) && input.value?.image_name === image_name)
    );

  const isIPAdapterImage = canvas.ipAdapters.entities.some(
    ({ ipAdapter }) => ipAdapter.image?.image_name === image_name
  );

  const imageUsage: ImageUsage = {
    isLayerImage: false,
    isNodesImage,
    isControlAdapterImage: false,
    isIPAdapterImage,
  };

  return imageUsage;
};

export const selectImageUsage = createMemoizedSelector(
  selectDeleteImageModalSlice,
  selectNodesSlice,
  selectCanvasSlice,
  (deleteImageModal, nodes, canvas) => {
    const { imagesToDelete } = deleteImageModal;

    if (!imagesToDelete.length) {
      return [];
    }

    const imagesUsage = imagesToDelete.map((i) => getImageUsage(nodes, canvas, i.image_name));

    return imagesUsage;
  }
);
