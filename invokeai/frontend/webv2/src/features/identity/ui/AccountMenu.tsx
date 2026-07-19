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

/**
 * The signed-in user's avatar menu — account settings, user management, and
 * sign-out — shown on both the Launchpad and the workbench shell. Renders only
 * in a multi-user session; the settings gear is a separate control that each
 * surface's top bar places alongside it.
 */
export const AccountMenu = () => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const navigate = useNavigate();
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  // User management is a Launchpad page (`/users`), not a workbench widget —
  // so this navigates there, working the same from Home and the editor.
  const openUserManagement = useCallback(() => {
    void navigate({ to: '/users' });
  }, [navigate]);

  const signOut = useCallback(async () => {
    await logoutSession();
    await navigate({ to: '/login' });
  }, [navigate]);

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
