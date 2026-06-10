import { HStack } from '@chakra-ui/react';
import { PiArrowClockwiseBold, PiArrowCounterClockwiseBold } from 'react-icons/pi';

import { IconButton } from '../../components/ui/Button';
import { useWorkbench } from '../../WorkbenchContext';

export const HistoryControlsWidgetView = () => {
  const { activeProject, dispatch } = useWorkbench();
  const canUndo = activeProject.undoRedo.past.length > 0;
  const canRedo = activeProject.undoRedo.future.length > 0;

  return (
    <HStack align="stretch" gap="1" h="full">
      <IconButton
        aria-label="Undo"
        aspectRatio="1 / 1"
        disabled={!canUndo}
        h="full"
        minH="0"
        minW="0"
        variant="ghost"
        onClick={(event) => {
          event.stopPropagation();
          dispatch({ type: 'undoProjectChange' });
        }}
      >
        <PiArrowCounterClockwiseBold />
      </IconButton>
      <IconButton
        aria-label="Redo"
        aspectRatio="1 / 1"
        disabled={!canRedo}
        h="full"
        minH="0"
        minW="0"
        variant="ghost"
        onClick={(event) => {
          event.stopPropagation();
          dispatch({ type: 'redoProjectChange' });
        }}
      >
        <PiArrowClockwiseBold />
      </IconButton>
    </HStack>
  );
};
