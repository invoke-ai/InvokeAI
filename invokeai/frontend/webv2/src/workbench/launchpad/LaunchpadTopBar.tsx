import type { BackendConnectionStatus } from '@platform/transport/types';

import { Box, Flex, HStack, Text } from '@chakra-ui/react';
import { AccountMenu } from '@features/identity';
import { useConnectionStatusSelector } from '@platform/transport/connectionStore';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { PaletteButton } from '@workbench/palette/PaletteButton';
import { SettingsButton } from '@workbench/settings/SettingsButton';
import { useTranslation } from 'react-i18next';

import { OpenProjectsControl } from './OpenProjectsControl';

const CONNECTION_LABEL_KEY: Record<Exclude<BackendConnectionStatus, 'connected'>, string> = {
  connecting: 'launchpad.connection.connecting',
  disconnected: 'launchpad.connection.disconnected',
};

/**
 * Subtle backend-connection indicator; hidden while the socket is healthy.
 *
 * The status changes on its own, with no interaction to attribute it to, so
 * the region is announced politely rather than only being visible.
 */
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
      <OpenProjectsControl />
    </HStack>
    <HStack gap="2">
      <ConnectionChip />
      <HStack gap="0.5">
        <PaletteButton />
        <SettingsButton />
        <AccountMenu />
      </HStack>
    </HStack>
  </Flex>
);
