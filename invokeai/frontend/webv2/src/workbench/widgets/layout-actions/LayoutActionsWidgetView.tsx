import { Button, HStack, Stack, Text } from '@chakra-ui/react';

import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';

export const LayoutActionsWidgetView = ({ presentation }: WidgetViewProps) => {
  const { dispatch } = useWorkbench();

  if (presentation === 'expanded') {
    return (
      <Stack align="start" gap="2">
        <Text fontSize="xs" fontWeight="700">
          Layout Actions
        </Text>
        <LayoutButtons
          onRecoverLayout={() => dispatch({ type: 'recoverShellLayout' })}
          onResetLayout={() => dispatch({ type: 'resetActiveLayout' })}
        />
        <Text color="fg.subtle" fontSize="2xs">
          Reset restores the active preset. Recover reopens the shell regions.
        </Text>
      </Stack>
    );
  }

  return (
    <LayoutButtons
      onRecoverLayout={() => dispatch({ type: 'recoverShellLayout' })}
      onResetLayout={() => dispatch({ type: 'resetActiveLayout' })}
    />
  );
};

const LayoutButtons = ({
  onRecoverLayout,
  onResetLayout,
}: {
  onRecoverLayout: () => void;
  onResetLayout: () => void;
}) => (
  <HStack gap="1">
    <Button
      color="fg.muted"
      fontSize="2xs"
      size="2xs"
      variant="ghost"
      _hover={{ color: 'fg.default' }}
      onClick={(event) => {
        event.stopPropagation();
        onResetLayout();
      }}
    >
      Reset Layout
    </Button>
    <Button
      color="fg.muted"
      fontSize="2xs"
      size="2xs"
      variant="ghost"
      _hover={{ color: 'fg.default' }}
      onClick={(event) => {
        event.stopPropagation();
        onRecoverLayout();
      }}
    >
      Recover Layout
    </Button>
  </HStack>
);
