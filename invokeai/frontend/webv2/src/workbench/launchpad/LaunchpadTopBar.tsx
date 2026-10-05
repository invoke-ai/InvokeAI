import type { BackendConnectionStatus } from '@platform/transport/types';

import { Badge, Box, Flex, HStack, Text } from '@chakra-ui/react';
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

  return (
    <Badge aria-live="polite" colorPalette={status === 'disconnected' ? 'red' : 'gray'} size="xl" variant="subtle">
      <Box aria-hidden bg="currentColor" boxSize="1.5" rounded="full" />
      {t(CONNECTION_LABEL_KEY[status])}
    </Badge>
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
      <Text fontSize="lg" fontWeight="700">
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
