import { Button, HStack } from '@chakra-ui/react';

import { useWorkbench } from '../../WorkbenchContext';

export const LayoutHeaderActions = () => {
  const { dispatch } = useWorkbench();

  return (
    <HStack gap="1">
      <Button
        color="fg.muted"
        fontSize="2xs"
        size="2xs"
        variant="ghost"
        _hover={{ color: 'fg.default' }}
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
        _hover={{ color: 'fg.default' }}
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
