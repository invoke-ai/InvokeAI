import { Button, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { PiWarningBold } from 'react-icons/pi';

import { useWorkbench } from '../WorkbenchContext';

/**
 * Copyable shell error surface.
 *
 * Phase 1 requires visible, copyable render/registration error details for
 * shell-level failures so they can be filed as issues. Widget-level isolation
 * arrives with the registry in Phase 3.
 */
export const ShellErrorSurface = () => {
  const { state } = useWorkbench();

  if (state.errorLog.length === 0) {
    return null;
  }

  return (
    <Stack
      bg="bg.surfaceRaised"
      borderWidth="1px"
      borderColor="red.500"
      bottom="9"
      gap="2"
      maxW="22rem"
      p="3"
      position="fixed"
      right="3"
      rounded="lg"
      shadow="lg"
      zIndex="overlay"
    >
      <HStack color="red.300" gap="1.5">
        <Icon as={PiWarningBold} boxSize="3.5" />
        <Text fontSize="xs" fontWeight="700">
          Shell error log
        </Text>
      </HStack>
      {state.errorLog.map((message, index) => (
        <HStack key={`${message}-${index}`} align="start" justify="space-between" gap="2">
          <Text color="fg.muted" fontSize="2xs">
            {message}
          </Text>
          <Button
            flexShrink={0}
            fontSize="2xs"
            size="2xs"
            variant="outline"
            borderColor="border.emphasis"
            onClick={() => void navigator.clipboard?.writeText(message)}
          >
            Copy
          </Button>
        </HStack>
      ))}
    </Stack>
  );
};
