import { Badge, Flex, IconButton, Menu, MenuButton, MenuItem, MenuList, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { pauseMediaCookieRefreshForLogout } from 'features/auth/hooks/useMediaCookieRefresh';
import { logout, selectCurrentUser } from 'features/auth/store/authSlice';
import { logoutAfterServerConfirmation } from 'features/auth/store/logoutAfterServerConfirmation';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearBold, PiSignOutBold, PiUserBold, PiUsersBold } from 'react-icons/pi';
import { useNavigate } from 'react-router-dom';
import { useLogoutMutation } from 'services/api/endpoints/auth';

export const UserMenu = memo(() => {
  const { t } = useTranslation();
  const user = useAppSelector(selectCurrentUser);
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const [logoutMutation] = useLogoutMutation();

  const handleLogout = useCallback(() => {
    void logoutAfterServerConfirmation(
      () => logoutMutation().unwrap(),
      () => {
        dispatch(logout());
        navigate('/login');
      },
      pauseMediaCookieRefreshForLogout
    ).catch(() => {
      // Local auth state is deliberately kept when the server can't confirm the logout
      // (the media cookie may still be live) — but silently doing nothing leaves the
      // user stuck with a dead button, so say why.
      toast({
        status: 'error',
        title: t('auth.logoutFailed'),
        description: t('auth.logoutFailedDesc'),
      });
    });
  }, [dispatch, navigate, logoutMutation, t]);

  const handleProfile = useCallback(() => {
    navigate('/profile');
  }, [navigate]);

  const handleUserManagement = useCallback(() => {
    navigate('/admin/users');
  }, [navigate]);

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
        <MenuItem icon={<PiGearBold />} onClick={handleProfile}>
          {t('auth.profile.menuItem')}
        </MenuItem>
        {user.is_admin && (
          <MenuItem icon={<PiUsersBold />} onClick={handleUserManagement}>
            {t('auth.userManagement.menuItem')}
          </MenuItem>
        )}
        <MenuItem icon={<PiSignOutBold />} onClick={handleLogout}>
          {t('auth.logout')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

UserMenu.displayName = 'UserMenu';
