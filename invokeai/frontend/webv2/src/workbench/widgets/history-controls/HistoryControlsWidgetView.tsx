import { Button, HStack, Stack, Text } from '@chakra-ui/react';
import { PiArrowClockwiseBold, PiArrowCounterClockwiseBold } from 'react-icons/pi';

import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';

export const HistoryControlsWidgetView = ({ presentation }: WidgetViewProps) => {
  const { activeProject, dispatch } = useWorkbench();
  const canUndo = activeProject.undoRedo.past.length > 0;
  const canRedo = activeProject.undoRedo.future.length > 0;

  if (presentation === 'expanded') {
    return (
      <Stack align="start" gap="2">
        <Text fontSize="xs" fontWeight="700">
          History Controls
        </Text>
        <HistoryButtons
          canRedo={canRedo}
          canUndo={canUndo}
          onRedo={() => dispatch({ type: 'redoProjectChange' })}
          onUndo={() => dispatch({ type: 'undoProjectChange' })}
        />
        <Text color="fg.subtle" fontSize="2xs">
          Undo entries: {activeProject.undoRedo.past.length}. Redo entries: {activeProject.undoRedo.future.length}.
        </Text>
      </Stack>
    );
  }

  return (
    <HistoryButtons
      canRedo={canRedo}
      canUndo={canUndo}
      onRedo={() => dispatch({ type: 'redoProjectChange' })}
      onUndo={() => dispatch({ type: 'undoProjectChange' })}
    />
  );
};

const HistoryButtons = ({
  canRedo,
  canUndo,
  onRedo,
  onUndo,
}: {
  canRedo: boolean;
  canUndo: boolean;
  onRedo: () => void;
  onUndo: () => void;
}) => (
  <HStack gap="1">
    <Button
      color="fg.muted"
      disabled={!canUndo}
      fontSize="2xs"
      size="2xs"
      variant="ghost"
      _hover={{ color: 'fg.default' }}
      onClick={(event) => {
        event.stopPropagation();
        onUndo();
      }}
    >
      <PiArrowCounterClockwiseBold /> Undo
    </Button>
    <Button
      color="fg.muted"
      disabled={!canRedo}
      fontSize="2xs"
      size="2xs"
      variant="ghost"
      _hover={{ color: 'fg.default' }}
      onClick={(event) => {
        event.stopPropagation();
        onRedo();
      }}
    >
      <PiArrowClockwiseBold /> Redo
    </Button>
  </HStack>
);
