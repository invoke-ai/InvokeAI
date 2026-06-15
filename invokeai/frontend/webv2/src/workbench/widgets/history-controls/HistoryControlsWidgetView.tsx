import { HStack } from '@chakra-ui/react';
import { IconButton } from '@workbench/components/ui/Button';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { Redo2Icon, Undo2Icon } from 'lucide-react';

export const HistoryControlsWidgetView = () => {
  const canUndo = useActiveProjectSelector((project) => project.undoRedo.past.length > 0);
  const canRedo = useActiveProjectSelector((project) => project.undoRedo.future.length > 0);
  const dispatch = useWorkbenchDispatch();

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
        <Undo2Icon />
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
        <Redo2Icon />
      </IconButton>
    </HStack>
  );
};
