import { HStack } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';

export const LayoutHeaderActions = () => {
  const dispatch = useWorkbenchDispatch();

  return (
    <HStack gap="1">
      <Button
        color="fg.muted"
        fontSize="2xs"
        size="2xs"
        variant="ghost"
        _hover={{ color: 'fg' }}
        onClick={(event) => {
          event.stopPropagation();
          dispatch({ type: 'resetActiveLayout' });
        }}
      >
        Reset Layout
      </Button>
      <Button
        color="fg.muted"
        fontSize="2xs"
        size="2xs"
        variant="ghost"
        _hover={{ color: 'fg' }}
        onClick={(event) => {
          event.stopPropagation();
          dispatch({ type: 'recoverShellLayout' });
        }}
      >
        Recover Layout
      </Button>
    </HStack>
  );
};
