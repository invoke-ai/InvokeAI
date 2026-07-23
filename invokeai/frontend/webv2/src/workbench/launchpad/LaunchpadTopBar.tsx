import type { BackendConnectionStatus } from '@platform/transport/types';

import { Box, Flex, HStack, Text } from '@chakra-ui/react';
import { AccountMenu } from '@features/identity';
import { useConnectionStatusSelector } from '@platform/transport/connectionStore';
import { Button } from '@platform/ui';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { Link, useSearch } from '@tanstack/react-router';
import { PaletteButton } from '@workbench/palette/PaletteButton';
import { SettingsButton } from '@workbench/settings/SettingsDialog';
import { ArrowLeftIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

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

export const LaunchpadTopBar = () => {
  const { t } = useTranslation();
  const search = useSearch({ strict: false }) as { project?: string };
  const projectSearch = useMemo(() => (search.project ? { project: search.project } : {}), [search.project]);

  return (
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
        <Button asChild size="xs" variant="subtle">
          <Link search={projectSearch} to="/app">
            <ArrowLeftIcon />
            {t('launchpad.backToProject')}
          </Link>
        </Button>
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
};
