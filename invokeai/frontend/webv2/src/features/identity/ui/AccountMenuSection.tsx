import { Avatar, Badge, Box, HStack, Icon, Menu, Stack } from '@chakra-ui/react';
import { logoutSession, useAuthSession } from '@features/identity/session';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useNavigate } from '@tanstack/react-router';
import { LogOutIcon, UserRoundCogIcon, UsersIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { ProfileDialog } from './ProfileDialog';

/**
 * Embed account actions directly in host menus to avoid nested popovers; render nothing in single-user mode. The host
 * renders {@link AccountProfileDialog} outside its menu: menu content unmounts on close, and would take the dialog along.
 */
export const AccountMenuSection = ({ onOpenProfile }: { onOpenProfile: () => void }) => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const navigate = useNavigate();

  const openUserManagement = useCallback(() => {
    void navigate({ to: '/users' });
  }, [navigate]);

  const signOut = useCallback(() => {
    void logoutSession();
  }, []);

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
            <Avatar.Root bg="accent.subtle" color="fg" size="sm">
              <Avatar.Fallback fontSize="xs" name={label} />
            </Avatar.Root>
            <Stack gap="0" minW="0">
              <MiddleTruncate fontSize="md" fontWeight="600" text={label} />
              <MiddleTruncate color="fg.muted" fontSize="xs" text={user.email} />
            </Stack>
          </HStack>
          {user.is_admin ? (
            <Badge colorPalette="purple" fontSize="xs" variant="surface">
              {t('users.admin')}
            </Badge>
          ) : null}
        </HStack>
      </Box>
      <Menu.Item value="account" onClick={onOpenProfile}>
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
    </>
  );
};

/** Account settings for the signed-in user, opened from {@link AccountMenuSection}; nothing once signed out. */
export const AccountProfileDialog = ({
  finalFocusEl,
  isOpen,
  onClose,
}: {
  /** The menu's trigger: the item that opened the dialog is gone once the menu closes. */
  finalFocusEl: () => HTMLElement | null;
  isOpen: boolean;
  onClose: () => void;
}) => {
  const session = useAuthSession();

  if (!session.multiuserEnabled || session.user === null) {
    return null;
  }

  return <ProfileDialog finalFocusEl={finalFocusEl} isOpen={isOpen} user={session.user} onClose={onClose} />;
};

/** True when {@link AccountMenuSection} has anything to render. */
export const useHasAccountSection = (): boolean => {
  const session = useAuthSession();

  return session.multiuserEnabled && session.user !== null;
};
