import { Badge, Flex, IconButton, Menu, MenuButton, MenuItem, MenuList, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { logout, selectCurrentUser } from 'features/auth/store/authSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSignOutBold, PiUserBold } from 'react-icons/pi';
import { useNavigate } from 'react-router-dom';
import { useLogoutMutation } from 'services/api/endpoints/auth';

export const UserMenu = memo(() => {
  const { t } = useTranslation();
  const user = useAppSelector(selectCurrentUser);
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const [logoutMutation] = useLogoutMutation();

  const handleLogout = useCallback(() => {
    // Call backend logout endpoint
    logoutMutation()
      .unwrap()
      .catch(() => {
        // Ignore errors - we'll log out locally anyway
      })
      .finally(() => {
        // Clear local state regardless of backend response
        dispatch(logout());
        navigate('/login');
      });
  }, [dispatch, navigate, logoutMutation]);

  if (!user) {
    return null;
  }

  return (
    <Menu>
      <Tooltip label={t('auth.userMenu')}>
        <MenuButton
          as={IconButton}
          aria-label={t('auth.userMenu')}
          icon={<PiUserBold />}
          variant="link"
          minW={8}
          w={8}
          h={8}
          borderRadius="base"
        />
      </Tooltip>
      <MenuList>
        <Flex px={3} py={2} flexDir="column" gap={1}>
          <Text fontSize="sm" fontWeight="semibold" noOfLines={1}>
            {user.display_name || user.email}
          </Text>
          <Text fontSize="xs" color="base.500" noOfLines={1}>
            {user.email}
          </Text>
          {user.is_admin && (
            <Badge colorScheme="invokeYellow" size="sm" alignSelf="flex-start" mt={1}>
              {t('auth.admin')}
            </Badge>
          )}
        </Flex>
        <MenuItem icon={<PiSignOutBold />} onClick={handleLogout}>
          {t('auth.logout')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

UserMenu.displayName = 'UserMenu';
