import { Avatar, Badge, Box, chakra, HStack, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { logoutSession, useAuthSession } from '@features/identity/session';
import { MenuContent } from '@platform/ui';
import { useNavigate } from '@tanstack/react-router';
import { ChevronDownIcon, LogOutIcon, UserRoundCogIcon, UsersIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ProfileDialog } from './ProfileDialog';

const MENU_POSITIONING = { placement: 'bottom-end' } as const;
const TRIGGER_HOVER = { bg: 'bg.subtle' } as const;

/** Show account actions only for multi-user sessions; each host places its separate settings control. */
export const AccountMenu = () => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const navigate = useNavigate();
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  const openUserManagement = useCallback(() => {
    void navigate({ to: '/users' });
  }, [navigate]);

  const signOut = useCallback(() => {
    void logoutSession();
  }, []);

  const openProfile = useCallback(() => setIsProfileOpen(true), []);
  const closeProfile = useCallback(() => setIsProfileOpen(false), []);

  if (!session.multiuserEnabled || session.user === null) {
    return;
  }

  const user = session.user;
  const label = user.display_name?.trim() || user.email;

  return (
    <>
      <Menu.Root positioning={MENU_POSITIONING}>
        <Menu.Trigger asChild>
          <chakra.button
            alignItems="center"
            color="fg"
            display="flex"
            flexShrink={0}
            gap="1.5"
            px="1.5"
            py="1"
            rounded="md"
            type="button"
            _hover={TRIGGER_HOVER}
          >
            <Avatar.Root bg="accent.subtle" color="fg" size="2xs">
              <Avatar.Fallback fontSize="2xs" name={label} />
            </Avatar.Root>
            <Text fontSize="xs" fontWeight="600">
              {label}
            </Text>
            <Icon as={ChevronDownIcon} boxSize="3" color="fg.muted" />
          </chakra.button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="56">
              <Box px="3" py="2">
                <HStack justify="space-between">
                  <Stack gap="0">
                    <Text fontSize="xs" fontWeight="600">
                      {label}
                    </Text>
                    <Text color="fg.muted" fontSize="2xs">
                      {user.email}
                    </Text>
                  </Stack>
                  {user.is_admin ? (
                    <Badge colorPalette="purple" fontSize="2xs" variant="surface">
                      {t('users.admin')}
                    </Badge>
                  ) : null}
                </HStack>
              </Box>
              <Menu.Separator />
              <Menu.Item value="account" onClick={openProfile}>
                <Icon as={UserRoundCogIcon} boxSize="3.5" />
                {t('auth.accountSettings')}
              </Menu.Item>
              {user.is_admin ? (
                <Menu.Item value="users" onClick={openUserManagement}>
                  <Icon as={UsersIcon} boxSize="3.5" />
                  {t('users.manageUsers')}
                </Menu.Item>
              ) : null}
              <Menu.Separator />
              <Menu.Item value="sign-out" onClick={signOut}>
                <Icon as={LogOutIcon} boxSize="3.5" />
                {t('auth.signOut')}
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ProfileDialog isOpen={isProfileOpen} user={user} onClose={closeProfile} />
    </>
  );
};
