import type { BackendConnectionStatus } from '@platform/transport/types';

import { Badge, Box, Flex, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { AccountMenu, useCapabilities } from '@features/identity';
import { useConnectionStatusSelector } from '@platform/transport/connectionStore';
import { IconButton } from '@platform/ui/Button';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { PaletteButton } from '@workbench/palette/PaletteButton';
import { DatabaseIcon } from 'lucide-react';
import { lazy, Suspense, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

const CONNECTION_LABEL_KEY: Record<Exclude<BackendConnectionStatus, 'connected'>, string> = {
  connecting: 'launchpad.connection.connecting',
  disconnected: 'launchpad.connection.disconnected',
};

const DATABASE_MENU_POSITIONING = { placement: 'bottom-end' } as const;
const LazyDatabaseMaintenanceDialog = lazy(() =>
  import('@workbench/shell/topbar/DatabaseMaintenanceDialog').then((module) => ({
    default: module.DatabaseMaintenanceDialog,
  }))
);

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
        <DatabaseMaintenanceMenu />
        <PaletteButton />
        <AccountMenu />
      </HStack>
    </HStack>
  </Flex>
);

export const DatabaseMaintenanceMenu = () => {
  const { t } = useTranslation();
  const { canManageAppConfig } = useCapabilities();
  const ids = useTooltipTriggerIds();
  const [isDialogMounted, setIsDialogMounted] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const handleMenuOpenChange = useCallback(({ open }: { open: boolean }) => setIsMenuOpen(open), []);
  const openConfirmation = useCallback(() => {
    setIsDialogMounted(true);
    setIsDialogOpen(true);
  }, []);
  const closeConfirmation = useCallback(() => setIsDialogOpen(false), []);

  if (!canManageAppConfig) {
    return null;
  }

  return (
    <>
      <Menu.Root ids={ids} positioning={DATABASE_MENU_POSITIONING} onOpenChange={handleMenuOpenChange}>
        <Tooltip content={t('settings.databaseMaintenance.menuLabel')} disabled={isMenuOpen} ids={ids} showArrow>
          <Menu.Trigger asChild>
            <IconButton aria-label={t('settings.databaseMaintenance.menuLabel')} size="lg" variant="ghost">
              <Icon as={DatabaseIcon} boxSize="4" />
            </IconButton>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="14rem">
              <Menu.Item value="run-vacuum" onClick={openConfirmation}>
                <Icon as={DatabaseIcon} boxSize="3.5" />
                <Menu.ItemText>{t('settings.databaseMaintenance.runVacuum')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      {isDialogMounted ? (
        <Suspense fallback={null}>
          <LazyDatabaseMaintenanceDialog isOpen={isDialogOpen} onClose={closeConfirmation} />
        </Suspense>
      ) : null}
    </>
  );
};
