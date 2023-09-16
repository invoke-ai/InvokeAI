import { Flex } from '@chakra-ui/react';
import IAIDropOverlay from 'common/components/IAIDropOverlay';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasToolbar from 'features/canvas/components/IAICanvasToolbar/IAICanvasToolbar';
import { useDroppableTypesafe } from 'features/dnd/hooks/typesafeHooks';
import { CanvasInitialImageDropData } from 'features/dnd/types';
import { isValidDrop } from 'features/dnd/util/isValidDrop';
import { memo } from 'react';

const droppableData: CanvasInitialImageDropData = {
  id: 'canvas-intial-image',
  actionType: 'SET_CANVAS_INITIAL_IMAGE',
};

const UnifiedCanvasContent = () => {
  const {
    isOver,
    setNodeRef: setDroppableRef,
    active,
  } = useDroppableTypesafe({
    id: 'unifiedCanvas',
    data: droppableData,
  });

  return (
    <Flex
      layerStyle="first"
      ref={setDroppableRef}
      tabIndex={-1}
      sx={{
        flexDirection: 'column',
        alignItems: 'center',
        gap: 4,
        p: 2,
        borderRadius: 'base',
        w: 'full',
        h: 'full',
      }}
    >
      <IAICanvasToolbar />
      <IAICanvas />
      {isValidDrop(droppableData, active) && (
        <IAIDropOverlay isOver={isOver} label="Set Canvas Initial Image" />
      )}
    </Flex>
  );
};

export default memo(UnifiedCanvasContent);
