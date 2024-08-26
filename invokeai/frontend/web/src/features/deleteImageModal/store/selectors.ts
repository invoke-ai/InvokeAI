import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectCanvasV2Slice } from 'features/controlLayers/store/selectors';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import { selectDeleteImageModalSlice } from 'features/deleteImageModal/store/slice';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { NodesState } from 'features/nodes/store/types';
import { isImageFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { some } from 'lodash-es';

import type { ImageUsage } from './types';
// TODO(psyche): handle image deletion (canvas sessions?)
export const getImageUsage = (nodes: NodesState, canvasV2: CanvasV2State, image_name: string) => {
  const isNodesImage = nodes.nodes
    .filter(isInvocationNode)
    .some((node) =>
      some(node.data.inputs, (input) => isImageFieldInputInstance(input) && input.value?.image_name === image_name)
    );

  const isIPAdapterImage = canvasV2.ipAdapters.entities.some(
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
  selectCanvasV2Slice,
  (deleteImageModal, nodes, canvasV2) => {
    const { imagesToDelete } = deleteImageModal;

    if (!imagesToDelete.length) {
      return [];
    }

    const imagesUsage = imagesToDelete.map((i) => getImageUsage(nodes, canvasV2, i.image_name));

    return imagesUsage;
  }
);
