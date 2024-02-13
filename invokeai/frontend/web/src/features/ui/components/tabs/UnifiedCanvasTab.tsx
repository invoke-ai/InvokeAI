import { Flex } from '@invoke-ai/ui-library';
import IAIDropOverlay from 'common/components/IAIDropOverlay';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasToolbar from 'features/canvas/components/IAICanvasToolbar/IAICanvasToolbar';
import { CANVAS_TAB_TESTID } from 'features/canvas/store/constants';
import { useDroppableTypesafe } from 'features/dnd/hooks/typesafeHooks';
import type { CanvasInitialImageDropData } from 'features/dnd/types';
import { isValidDrop } from 'features/dnd/util/isValidDrop';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const droppableData: CanvasInitialImageDropData = {
  id: 'canvas-intial-image',
  actionType: 'SET_CANVAS_INITIAL_IMAGE',
};

const UnifiedCanvasTab = () => {
  const { t } = useTranslation();
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
      flexDirection="column"
      alignItems="center"
      gap={4}
      p={2}
      borderRadius="base"
      w="full"
      h="full"
      data-testid={CANVAS_TAB_TESTID}
    >
      <IAICanvasToolbar />
      <IAICanvas />
      {isValidDrop(droppableData, active) && (
        <IAIDropOverlay isOver={isOver} label={t('toast.setCanvasInitialImage')} />
      )}
    </Flex>
  );
};

export default memo(UnifiedCanvasTab);
