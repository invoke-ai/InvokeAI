import { Flex } from '@invoke-ai/ui-library';
import IAIDroppable from 'common/components/IAIDroppable';
import type { AddLayerFromImageDropData } from 'features/dnd/types';
import { memo } from 'react';

const addLayerFromImageDropData: AddLayerFromImageDropData = {
  id: 'add-layer-from-image-drop-data',
  actionType: 'ADD_LAYER_FROM_IMAGE',
};

export const CanvasDropArea = memo(() => {
  return (
    <Flex position="absolute" top={0} right={0} bottom={0} left={0} gap={2} pointerEvents="none">
      <IAIDroppable dropLabel="Create Layer" data={addLayerFromImageDropData} />
    </Flex>
  );
});

CanvasDropArea.displayName = 'CanvasDropArea';
