import { Flex } from '@invoke-ai/ui-library';
import IAIDroppable from 'common/components/IAIDroppable';
import type { AddControlLayerFromImageDropData, AddRasterLayerFromImageDropData } from 'features/dnd/types';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo } from 'react';

const addRasterLayerFromImageDropData: AddRasterLayerFromImageDropData = {
  id: 'add-raster-layer-from-image-drop-data',
  actionType: 'ADD_RASTER_LAYER_FROM_IMAGE',
};

const addControlLayerFromImageDropData: AddControlLayerFromImageDropData = {
  id: 'add-control-layer-from-image-drop-data',
  actionType: 'ADD_CONTROL_LAYER_FROM_IMAGE',
};

export const CanvasDropArea = memo(() => {
  const imageViewer = useImageViewer();

  if (imageViewer.isOpen) {
    return null;
  }

  return (
    <>
      <Flex position="absolute" top={0} right={0} bottom="50%" left={0} gap={2} pointerEvents="none">
        <IAIDroppable dropLabel="Create Raster Layer" data={addRasterLayerFromImageDropData} />
      </Flex>
      <Flex position="absolute" top="50%" right={0} bottom={0} left={0} gap={2} pointerEvents="none">
        <IAIDroppable dropLabel="Create Control Layer" data={addControlLayerFromImageDropData} />
      </Flex>
    </>
  );
});

CanvasDropArea.displayName = 'CanvasDropArea';
