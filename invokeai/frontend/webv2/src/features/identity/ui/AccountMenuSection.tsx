import { Avatar, Badge, Box, HStack, Icon, Menu, Stack, Text } from '@chakra-ui/react';
import { logoutSession, useAuthSession } from '@features/identity/session';
import { useNavigate } from '@tanstack/react-router';
import { LogOutIcon, UserRoundCogIcon, UsersIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ProfileDialog } from './ProfileDialog';

/**
 * The signed-in user's rows — identity header, account settings, user
 * management, sign-out — as menu items rather than a menu of their own, so a
 * host menu can embed them instead of nesting a second popover inside itself.
 *
 * Renders nothing in a single-user session, where there is no account to manage.
 */
export const AccountMenuSection = () => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const navigate = useNavigate();
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  // User management is a Launchpad page (`/users`), not a workbench widget —
  // so this navigates there, working the same from Home and the editor.
  const openUserManagement = useCallback(() => {
    void navigate({ to: '/users' });
  }, [navigate]);

  const signOut = useCallback(() => {
    void logoutSession();
  }, []);

  const openProfile = useCallback(() => setIsProfileOpen(true), []);
  const closeProfile = useCallback(() => setIsProfileOpen(false), []);

  if (!session.multiuserEnabled || session.user === null) {
    return null;
  }

  const user = session.user;
  const label = user.display_name?.trim() || user.email;

  return (
    <>
      <Box px="3" py="2">
        <HStack justify="space-between">
          <HStack gap="2" minW="0">
            <Avatar.Root bg="accent.subtle" color="fg" size="2xs">
              <Avatar.Fallback fontSize="2xs" name={label} />
            </Avatar.Root>
            <Stack gap="0" minW="0">
              <Text fontSize="xs" fontWeight="600" truncate>
                {label}
              </Text>
              <Text color="fg.muted" fontSize="2xs" truncate>
                {user.email}
              </Text>
            </Stack>
          </HStack>
          {user.is_admin ? (
            <Badge colorPalette="purple" fontSize="2xs" variant="surface">
              {t('users.admin')}
            </Badge>
          ) : null}
        </HStack>
      </Box>
      <Menu.Item value="account" onClick={openProfile}>
        <Icon as={UserRoundCogIcon} boxSize="3.5" />
        <Menu.ItemText>{t('auth.accountSettings')}</Menu.ItemText>
      </Menu.Item>
      {user.is_admin ? (
        <Menu.Item value="users" onClick={openUserManagement}>
          <Icon as={UsersIcon} boxSize="3.5" />
          <Menu.ItemText>{t('users.manageUsers')}</Menu.ItemText>
        </Menu.Item>
      ) : null}
      <Menu.Item value="sign-out" onClick={signOut}>
        <Icon as={LogOutIcon} boxSize="3.5" />
        <Menu.ItemText>{t('auth.signOut')}</Menu.ItemText>
      </Menu.Item>
      <ProfileDialog isOpen={isProfileOpen} user={user} onClose={closeProfile} />
    </>
  );
};

/** True when {@link AccountMenuSection} has anything to render. */
export const useHasAccountSection = (): boolean => {
  const session = useAuthSession();

  return session.multiuserEnabled && session.user !== null;
};
