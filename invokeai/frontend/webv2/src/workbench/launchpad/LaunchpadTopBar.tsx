import type { BackendConnectionStatus } from '@workbench/types';

import { Box, Flex, HStack, Text } from '@chakra-ui/react';
import { AccountMenu } from '@workbench/auth/components/AccountMenu';
import { useConnectionStatusSelector } from '@workbench/backend/connectionStore';
import { InvokeMark } from '@workbench/components/InvokeMark';
import { SettingsButton } from '@workbench/settings/SettingsDialog';

const CONNECTION_LABEL: Record<Exclude<BackendConnectionStatus, 'connected'>, string> = {
  connecting: 'Connecting…',
  disconnected: 'Disconnected',
};

/** Subtle backend-connection indicator; hidden while the socket is healthy. */
const ConnectionChip = () => {
  const status = useConnectionStatusSelector((snapshot) => snapshot.status);

  if (status === 'connected') {
    return null;
  }

  const isDown = status === 'disconnected';

  return (
    <HStack
      bg="bg.muted"
      borderColor={isDown ? 'border.error' : 'border.subtle'}
      borderWidth="1px"
      gap="1.5"
      px="2.5"
      py="1"
      rounded="full"
    >
      <Box bg={isDown ? 'fg.error' : 'fg.muted'} boxSize="1.5" rounded="full" />
      <Text color={isDown ? 'fg.error' : 'fg.muted'} fontSize="2xs" fontWeight="600">
        {CONNECTION_LABEL[status]}
      </Text>
    </HStack>
  );
};

export const LaunchpadTopBar = () => (
  <Flex
    align="center"
    borderBottomWidth="1px"
    bg="bg.subtle"
    flexShrink={0}
    justify="space-between"
    pe="1.5"
    ps="4"
    h="12"
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
        <SettingsButton />
        <AccountMenu />
      </HStack>
    </HStack>
  </Flex>
);
