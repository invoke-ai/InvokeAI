import type { SystemStyleObject } from '@chakra-ui/react';

import { Avatar, Badge, Box, HStack, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { logoutSession, useAuthSession } from '@features/identity/session';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { MiddleTruncate, type MiddleTruncateProps } from '@platform/ui/MiddleTruncate';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useNavigate } from '@tanstack/react-router';
import { ChevronRightIcon, LogOutIcon, UserRoundCogIcon, UsersIcon } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ProfileDialog } from './ProfileDialog';

// Opens beside a navigation rail's footer, rising from its bottom edge, so it never runs off the window.
const MENU_POSITIONING = { placement: 'right-end' } as const;
/** Two lines need more than a rail link's fixed height; the padding keeps the avatar clear of the hover fill's edge. */
const ROW_TRIGGER: SystemStyleObject = { h: 'auto', justifyContent: 'start', px: '2', py: '1.5' };
/**
 * Trimmed to the capitals' height and baseline, the initials' box is their ink, so the circle's flex centring centres
 * the letters themselves; untrimmed, the font's descender room sits below them and lifts them off centre. Browsers
 * without `text-box` keep that small lift.
 */
const INITIALS: SystemStyleObject = { textBox: 'trim-both cap alphabetic' };

/** 28px: the rail's 32px items with a 2px inset, in both widths. */
/** Decorative: the name beside it, or the icon-only button's label, names the account. */
const AccountAvatar = ({ label }: { label: string }) => (
  <Avatar.Root aria-hidden bg="accent.subtle" boxSize="7" color="fg" flexShrink={0}>
    <Avatar.Fallback css={INITIALS} fontSize="sm" fontWeight="600" name={label} />
  </Avatar.Root>
);

/** An email keeps its domain when it truncates, as the menu header shows it; names and roles truncate at the end. */
const AccountLine = ({ isEmail, text, ...props }: Omit<MiddleTruncateProps, 'tailGraphemes'> & { isEmail: boolean }) =>
  isEmail ? (
    <MiddleTruncate {...props} text={text} />
  ) : (
    <Text {...props} truncate>
      {text}
    </Text>
  );

/** True in multi-user mode with someone signed in: when {@link AccountMenu} renders anything. */
export const useHasAccountMenu = (): boolean => {
  const session = useAuthSession();

  return session.multiuserEnabled && session.user !== null;
};

/**
 * The signed-in account and its actions, for the foot of a navigation rail; nothing in single-user mode. `compact` is
 * the icon-only rail: the trigger shows the avatar, named by a tooltip.
 */
export const AccountMenu = ({ compact = false }: { compact?: boolean }) => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const navigate = useNavigate();
  const ids = useTooltipTriggerIds();
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  const openUserManagement = useCallback(() => {
    void navigate({ to: '/users' });
  }, [navigate]);

  const signOut = useCallback(() => {
    void logoutSession();
  }, []);

  const openProfile = useCallback(() => setIsProfileOpen(true), []);
  const closeProfile = useCallback(() => setIsProfileOpen(false), []);
  // The menu item that opened the dialog is gone by the time it closes; focus returns to the menu's own button.
  const triggerRef = useRef<HTMLButtonElement>(null);
  const getTrigger = useCallback(() => triggerRef.current, []);

  if (!session.multiuserEnabled || session.user === null) {
    return;
  }

  const user = session.user;
  const label = user.display_name?.trim() || user.email;
  // The email when the name is a display name; an email-only account shows its role instead of repeating itself.
  const detail = label === user.email ? (user.is_admin ? t('users.admin') : t('users.user')) : user.email;
  const triggerLabel = t('auth.accountMenu', { name: label });

  return (
    <>
      <Menu.Root ids={ids} positioning={MENU_POSITIONING}>
        {compact ? (
          <Tooltip content={label} ids={ids} placement="right">
            <Menu.Trigger asChild>
              <IconButton ref={triggerRef} aria-label={triggerLabel} size="lg" variant="ghost">
                <AccountAvatar label={label} />
              </IconButton>
            </Menu.Trigger>
          </Tooltip>
        ) : (
          <Menu.Trigger asChild>
            {/* Named by its visible lines, so the spoken name matches what is shown. */}
            <Button ref={triggerRef} css={ROW_TRIGGER} gap="2.5" variant="ghost" w="full">
              <AccountAvatar label={label} />
              <Stack flex="1" gap="0" minW="0" textAlign="start">
                <AccountLine fontSize="md" fontWeight="600" isEmail={label === user.email} text={label} />
                <AccountLine
                  color="fg.muted"
                  fontSize="xs"
                  fontWeight="400"
                  isEmail={detail === user.email}
                  text={detail}
                />
              </Stack>
              <Icon as={ChevronRightIcon} boxSize="3" color="fg.muted" flexShrink={0} />
            </Button>
          </Menu.Trigger>
        )}
        <Portal>
          <Menu.Positioner>
            <MenuContent maxW="20rem" minW="56">
              <Box px="3" py="2">
                <HStack gap="2" justify="space-between">
                  <Stack gap="0" minW="0">
                    <MiddleTruncate fontSize="md" fontWeight="600" text={label} />
                    <MiddleTruncate color="fg.muted" fontSize="xs" text={user.email} />
                  </Stack>
                  {user.is_admin ? (
                    <Badge colorPalette="purple" flexShrink={0} fontSize="xs" variant="surface">
                      {t('users.admin')}
                    </Badge>
                  ) : null}
                </HStack>
              </Box>
              <Menu.Separator />
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
              <Menu.Separator />
              <Menu.Item value="sign-out" onClick={signOut}>
                <Icon as={LogOutIcon} boxSize="3.5" />
                <Menu.ItemText>{t('auth.signOut')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ProfileDialog finalFocusEl={getTrigger} isOpen={isProfileOpen} user={user} onClose={closeProfile} />
    </>
  );
};
