import { Avatar, chakra, Flex, HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { useNavigate } from '@tanstack/react-router';
import { ChevronDownIcon, LogOutIcon } from 'lucide-react';

import { InvokeMark } from '../auth/components/AuthScreen';
import { MenuContent } from '../components/ui/Menu';
import { logoutSession, useAuthSession } from '../auth/session';

/**
 * Home's slim header: brand on the left, the signed-in account on the right.
 * Account settings and user management live in the editor shell; here the
 * menu only carries identity and sign-out, and single-user mode shows no
 * account cluster at all.
 */
export const HomeTopBar = () => (
  <Flex
    align="center"
    borderBottomWidth="1px"
    borderColor="border.subtle"
    flexShrink={0}
    justify="space-between"
    px={{ base: '5', md: '8' }}
    py="3"
  >
    <HStack gap="2.5">
      <InvokeMark size={22} />
      <Text fontSize="sm" fontWeight="700">
        Invoke
      </Text>
    </HStack>
    <HomeUserMenu />
  </Flex>
);

const HomeUserMenu = () => {
  const session = useAuthSession();
  const navigate = useNavigate();

  if (!session.multiuserEnabled || session.user === null) {
    return null;
  }

  const user = session.user;
  const label = user.display_name?.trim() || user.email;

  const signOut = async () => {
    await logoutSession();
    await navigate({ to: '/login' });
  };

  return (
    <Menu.Root positioning={{ placement: 'bottom-end' }}>
      <Menu.Trigger asChild>
        <chakra.button
          alignItems="center"
          color="fg.default"
          display="flex"
          flexShrink={0}
          gap="1.5"
          px="1.5"
          py="1"
          rounded="md"
          type="button"
          _hover={{ bg: 'bg.surface' }}
        >
          <Avatar.Root bg="accent.widget" color="fg.default" size="2xs">
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
          <MenuContent minW="48">
            <Text color="fg.muted" fontSize="2xs" px="3" py="2">
              {user.email}
            </Text>
            <Menu.Separator />
            <Menu.Item value="sign-out" onClick={() => void signOut()}>
              <Icon as={LogOutIcon} boxSize="3.5" />
              Sign out
            </Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
