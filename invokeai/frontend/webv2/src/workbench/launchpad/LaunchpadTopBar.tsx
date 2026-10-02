import type { BackendConnectionStatus } from '@platform/transport/types';

import { Box, Flex, HStack, Text } from '@chakra-ui/react';
import { AccountMenu } from '@features/identity';
import { useConnectionStatusSelector } from '@platform/transport/connectionStore';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { PaletteButton } from '@workbench/palette/PaletteButton';
import { useTranslation } from 'react-i18next';

const CONNECTION_LABEL_KEY: Record<Exclude<BackendConnectionStatus, 'connected'>, string> = {
  connecting: 'launchpad.connection.connecting',
  disconnected: 'launchpad.connection.disconnected',
};

/** Hide healthy connection status; announce spontaneous status changes politely. */
const ConnectionChip = () => {
  const { t } = useTranslation();
  const status = useConnectionStatusSelector((snapshot) => snapshot.status);

  if (status === 'connected') {
    return null;
  }

  const isDown = status === 'disconnected';

  return (
    <HStack
      aria-live="polite"
      bg="bg.muted"
      borderColor={isDown ? 'border.error' : 'border.subtle'}
      borderWidth="1px"
      gap="1.5"
      px="2.5"
      py="1"
      rounded="full"
    >
      <Box aria-hidden bg={isDown ? 'fg.error' : 'fg.muted'} boxSize="1.5" rounded="full" />
      <Text color={isDown ? 'fg.error' : 'fg.muted'} fontSize="2xs" fontWeight="600">
        {t(CONNECTION_LABEL_KEY[status])}
      </Text>
    </HStack>
  );
};

export const LaunchpadTopBar = () => (
  <Flex
    align="center"
    bg="bg.subtle"
    borderBottomWidth="1px"
    flexShrink={0}
    h="12"
    justify="space-between"
    pe="1.5"
    ps="4"
  >
    <HStack gap="3">
      <InvokeMark size={20} />
      <Text fontSize="sm" fontWeight="700">
        Invoke
      </Text>
    </HStack>
    <HStack gap="2">
      <ConnectionChip />
      <HStack gap="0.5">
        <PaletteButton />
        <AccountMenu />
      </HStack>
    </HStack>
  </Flex>
);
