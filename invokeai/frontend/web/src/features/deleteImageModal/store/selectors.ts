import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import type { CanvasState } from 'features/canvas/store/canvasTypes';
import {
  selectControlAdapterAll,
  selectControlAdaptersSlice,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdaptersState } from 'features/controlAdapters/store/types';
import { isControlNetOrT2IAdapter } from 'features/controlAdapters/store/types';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import type { CanvasV2State } from 'features/controlLayers/store/types';
import {
  isControlAdapterLayer,
  isInitialImageLayer,
  isIPAdapterLayer,
  isRegionalGuidanceLayer,
} from 'features/controlLayers/store/types';
import { selectDeleteImageModalSlice } from 'features/deleteImageModal/store/slice';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { NodesState } from 'features/nodes/store/types';
import { isImageFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { some } from 'lodash-es';

import type { ImageUsage } from './types';

export const getImageUsage = (
  canvas: CanvasState,
  nodes: NodesState,
  controlAdapters: ControlAdaptersState,
  controlLayers: CanvasV2State,
  image_name: string
) => {
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

  const isControlLayerImage = controlLayers.layers.some((l) => {
    if (isRegionalGuidanceLayer(l)) {
      return l.ipAdapters.some((ipa) => ipa.image?.name === image_name);
    }
    if (isControlAdapterLayer(l)) {
      return l.controlAdapter.image?.name === image_name || l.controlAdapter.processedImage?.name === image_name;
    }
    if (isIPAdapterLayer(l)) {
      return l.ipAdapter.image?.name === image_name;
    }
    if (isInitialImageLayer(l)) {
      return l.image?.name === image_name;
    }
    return false;
  });

  const imageUsage: ImageUsage = {
    isCanvasImage,
    isNodesImage,
    isControlImage,
    isControlLayerImage,
  };

  return imageUsage;
};

export const selectImageUsage = createMemoizedSelector(
  selectDeleteImageModalSlice,
  selectCanvasSlice,
  selectNodesSlice,
  selectControlAdaptersSlice,
  selectCanvasV2Slice,
  (deleteImageModal, canvas, nodes, controlAdapters, controlLayers) => {
    const { imagesToDelete } = deleteImageModal;

    if (!imagesToDelete.length) {
      return [];
    }

    const imagesUsage = imagesToDelete.map((i) =>
      getImageUsage(canvas, nodes, controlAdapters, canvasV2, i.image_name)
    );

    return imagesUsage;
  }
);
