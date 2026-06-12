import { Avatar, Badge, Box, chakra, HStack, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { useNavigate } from '@tanstack/react-router';
import { useState } from 'react';
import { ChevronDownIcon, LogOutIcon, UserRoundCogIcon, UsersIcon } from 'lucide-react';

import { MenuContent } from '../../components/ui/Menu';
import { SettingsButton } from '../../settings/SettingsDialog';
import { useOptionalWorkbench } from '../../WorkbenchContext';
import { logoutSession, useAuthSession } from '../session';
import { ProfileDialog } from './ProfileDialog';

/**
 * Shared settings/account cluster used by both Home and the workbench shell:
 * the settings gear always renders; the avatar menu joins it when a
 * multi-user session is signed in.
 */
export const AccountMenu = () => {
  const session = useAuthSession();
  const workbench = useOptionalWorkbench();
  const navigate = useNavigate();
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  if (!session.multiuserEnabled || session.user === null) {
    return <SettingsButton />;
  }

  const user = session.user;
  const label = user.display_name?.trim() || user.email;

  const openUserManagement = () => {
    const activeProject = workbench?.activeProject;

    if (!activeProject || !workbench) {
      return;
    }

    const centerRegion = activeProject.widgetRegions.center;

    if (centerRegion.enabledWidgetIds.includes('users')) {
      workbench.dispatch({ region: 'center', type: 'selectRegionWidget', widgetId: 'users' });
    } else {
      workbench.dispatch({ region: 'center', type: 'toggleRegionWidget', widgetId: 'users' });
    }
  };

  const signOut = async () => {
    await logoutSession();
    await navigate({ to: '/login' });
  };

  return (
    <HStack gap="0.5">
      <SettingsButton />
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
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
            _hover={{ bg: 'bg.subtle' }}
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
                      Admin
                    </Badge>
                  ) : null}
                </HStack>
              </Box>
              <Menu.Separator />
              <Menu.Item value="account" onClick={() => setIsProfileOpen(true)}>
                <Icon as={UserRoundCogIcon} boxSize="3.5" />
                Account settings
              </Menu.Item>
              {user.is_admin && workbench ? (
                <Menu.Item value="users" onClick={openUserManagement}>
                  <Icon as={UsersIcon} boxSize="3.5" />
                  Manage users
                </Menu.Item>
              ) : null}
              <Menu.Separator />
              <Menu.Item value="sign-out" onClick={() => void signOut()}>
                <Icon as={LogOutIcon} boxSize="3.5" />
                Sign out
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ProfileDialog isOpen={isProfileOpen} user={user} onClose={() => setIsProfileOpen(false)} />
    </HStack>
  );
};
