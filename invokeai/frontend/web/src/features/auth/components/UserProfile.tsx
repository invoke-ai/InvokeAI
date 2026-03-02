import {
  Box,
  Button,
  Center,
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  Grid,
  GridItem,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spinner,
  Text,
  Tooltip,
  VStack,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectAuthToken, selectCurrentUser, setCredentials } from 'features/auth/store/authSlice';
import type { ChangeEvent, FormEvent } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiEyeSlashBold, PiLightningFill } from 'react-icons/pi';
import { useNavigate } from 'react-router-dom';
import { useLazyGeneratePasswordQuery, useUpdateCurrentUserMutation } from 'services/api/endpoints/auth';

const validatePasswordStrength = (
  password: string,
  t: (key: string) => string
): { isValid: boolean; message: string } => {
  if (password.length === 0) {
    return { isValid: true, message: '' };
  }
  if (password.length < 8) {
    return { isValid: false, message: t('auth.setup.passwordTooShort') };
  }
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);
  if (!hasUpper || !hasLower || !hasDigit) {
    return { isValid: false, message: t('auth.setup.passwordMissingRequirements') };
  }
  return { isValid: true, message: '' };
};

const PASSWORD_GRID_COLUMNS = '180px 1fr';

export const UserProfile = memo(() => {
  const { t } = useTranslation();
  const currentUser = useAppSelector(selectCurrentUser);
  const currentToken = useAppSelector(selectAuthToken);
  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  const [displayName, setDisplayName] = useState(currentUser?.display_name ?? '');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [updateCurrentUser, { isLoading }] = useUpdateCurrentUserMutation();
  const [triggerGeneratePassword] = useLazyGeneratePasswordQuery();

  const newPasswordValidation = validatePasswordStrength(newPassword, t);

  const isPasswordChangeAttempted = newPassword.length > 0 || currentPassword.length > 0;
  const passwordsMatch = newPassword.length > 0 && newPassword === confirmPassword;
  const isPasswordChangeValid =
    !isPasswordChangeAttempted || (currentPassword.length > 0 && newPasswordValidation.isValid && passwordsMatch);

  const handleCancel = useCallback(() => {
    navigate(-1);
  }, [navigate]);

  const handleGeneratePassword = useCallback(async () => {
    try {
      const result = await triggerGeneratePassword().unwrap();
      setNewPassword(result.password);
      setConfirmPassword(result.password);
      setShowNewPassword(true);
      setShowConfirmPassword(true);
    } catch {
      // ignore
    }
  }, [triggerGeneratePassword]);

  const toggleShowCurrentPassword = useCallback(() => {
    setShowCurrentPassword((v) => !v);
  }, []);

  const toggleShowNewPassword = useCallback(() => {
    setShowNewPassword((v) => !v);
  }, []);

  const toggleShowConfirmPassword = useCallback(() => {
    setShowConfirmPassword((v) => !v);
  }, []);

  const handleDisplayNameChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setDisplayName(e.target.value);
  }, []);

  const handleCurrentPasswordChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setCurrentPassword(e.target.value);
  }, []);

  const handleNewPasswordChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setNewPassword(e.target.value);
  }, []);

  const handleConfirmPasswordChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setConfirmPassword(e.target.value);
  }, []);

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      setErrorMessage(null);

      if (!isPasswordChangeValid) {
        return;
      }

      try {
        const updatePayload: Parameters<typeof updateCurrentUser>[0] = {
          display_name: displayName || null,
        };
        if (newPassword) {
          updatePayload.current_password = currentPassword;
          updatePayload.new_password = newPassword;
        }
        const updatedUser = await updateCurrentUser(updatePayload).unwrap();

        // Refresh the stored user info so the header reflects the new display name
        if (currentToken) {
          dispatch(
            setCredentials({
              token: currentToken,
              user: {
                user_id: updatedUser.user_id,
                email: updatedUser.email,
                display_name: updatedUser.display_name ?? null,
                is_admin: updatedUser.is_admin ?? false,
                is_active: updatedUser.is_active ?? true,
              },
            })
          );
        }

        // Navigate back after successful save
        navigate(-1);
      } catch (err) {
        const detail =
          err && typeof err === 'object' && 'data' in err && typeof (err as { data: unknown }).data === 'object'
            ? ((err as { data: { detail?: string } }).data?.detail ?? t('auth.profile.saveFailed'))
            : t('auth.profile.saveFailed');
        setErrorMessage(detail);
      }
    },
    [
      displayName,
      currentPassword,
      newPassword,
      isPasswordChangeValid,
      updateCurrentUser,
      currentToken,
      dispatch,
      navigate,
      t,
    ]
  );

  if (!currentUser) {
    return (
      <Center py={12}>
        <Spinner size="xl" />
      </Center>
    );
  }

  return (
    <Box p={6} maxW="480px">
      <Heading size="md" mb={6}>
        {t('auth.profile.title')}
      </Heading>

      <form onSubmit={handleSubmit}>
        <VStack spacing={5} align="stretch">
          {/* Email (read-only) */}
          <FormControl>
            <FormLabel>{t('auth.profile.email')}</FormLabel>
            <Input type="email" value={currentUser.email} isReadOnly opacity={0.6} />
            <FormHelperText>{t('auth.profile.emailReadOnly')}</FormHelperText>
          </FormControl>

          {/* Display name */}
          <FormControl>
            <FormLabel>{t('auth.profile.displayName')}</FormLabel>
            <Input
              type="text"
              value={displayName}
              onChange={handleDisplayNameChange}
              placeholder={t('auth.profile.displayNamePlaceholder')}
            />
          </FormControl>

          <Box borderTop="1px solid" borderColor="base.600" pt={4}>
            <Text fontSize="sm" fontWeight="semibold" mb={4} color="base.300">
              {t('auth.profile.changePassword')}
            </Text>

            {/* Current password */}
            <FormControl mb={4} isRequired={newPassword.length > 0}>
              <Grid templateColumns={PASSWORD_GRID_COLUMNS} gap={4} alignItems="start">
                <GridItem>
                  <FormLabel textAlign="right" mb={0} pt={2}>
                    {t('auth.profile.currentPassword')}
                  </FormLabel>
                </GridItem>
                <GridItem>
                  <InputGroup>
                    <Input
                      type={showCurrentPassword ? 'text' : 'password'}
                      value={currentPassword}
                      onChange={handleCurrentPasswordChange}
                      placeholder={t('auth.profile.currentPasswordPlaceholder')}
                      autoComplete="current-password"
                      pr="3rem"
                    />
                    <InputRightElement>
                      <Tooltip
                        label={
                          showCurrentPassword
                            ? t('auth.userManagement.hidePassword')
                            : t('auth.userManagement.showPassword')
                        }
                      >
                        <IconButton
                          aria-label={
                            showCurrentPassword
                              ? t('auth.userManagement.hidePassword')
                              : t('auth.userManagement.showPassword')
                          }
                          icon={showCurrentPassword ? <PiEyeSlashBold /> : <PiEyeBold />}
                          variant="ghost"
                          size="sm"
                          onClick={toggleShowCurrentPassword}
                          tabIndex={-1}
                        />
                      </Tooltip>
                    </InputRightElement>
                  </InputGroup>
                </GridItem>
              </Grid>
            </FormControl>

            {/* New password */}
            <FormControl isInvalid={newPassword.length > 0 && !newPasswordValidation.isValid} mb={4}>
              <Grid templateColumns={PASSWORD_GRID_COLUMNS} gap={4} alignItems="start">
                <GridItem>
                  <FormLabel textAlign="right" mb={0} pt={2}>
                    {t('auth.profile.newPassword')}
                  </FormLabel>
                </GridItem>
                <GridItem>
                  <InputGroup>
                    <Input
                      type={showNewPassword ? 'text' : 'password'}
                      value={newPassword}
                      onChange={handleNewPasswordChange}
                      placeholder={t('auth.profile.newPasswordPlaceholder')}
                      autoComplete="new-password"
                      pr="3rem"
                    />
                    <InputRightElement>
                      <Tooltip
                        label={
                          showNewPassword
                            ? t('auth.userManagement.hidePassword')
                            : t('auth.userManagement.showPassword')
                        }
                      >
                        <IconButton
                          aria-label={
                            showNewPassword
                              ? t('auth.userManagement.hidePassword')
                              : t('auth.userManagement.showPassword')
                          }
                          icon={showNewPassword ? <PiEyeSlashBold /> : <PiEyeBold />}
                          variant="ghost"
                          size="sm"
                          onClick={toggleShowNewPassword}
                          tabIndex={-1}
                        />
                      </Tooltip>
                    </InputRightElement>
                  </InputGroup>
                  {newPassword.length > 0 && !newPasswordValidation.isValid && (
                    <FormErrorMessage>{newPasswordValidation.message}</FormErrorMessage>
                  )}
                </GridItem>
              </Grid>
            </FormControl>

            {/* Confirm new password */}
            <FormControl isInvalid={confirmPassword.length > 0 && !passwordsMatch} mb={4}>
              <Grid templateColumns={PASSWORD_GRID_COLUMNS} gap={4} alignItems="start">
                <GridItem>
                  <FormLabel textAlign="right" mb={0} pt={2}>
                    {t('auth.profile.confirmPassword')}
                  </FormLabel>
                </GridItem>
                <GridItem>
                  <InputGroup>
                    <Input
                      type={showConfirmPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={handleConfirmPasswordChange}
                      placeholder={t('auth.profile.confirmPasswordPlaceholder')}
                      autoComplete="new-password"
                      pr="3rem"
                    />
                    <InputRightElement>
                      <Tooltip
                        label={
                          showConfirmPassword
                            ? t('auth.userManagement.hidePassword')
                            : t('auth.userManagement.showPassword')
                        }
                      >
                        <IconButton
                          aria-label={
                            showConfirmPassword
                              ? t('auth.userManagement.hidePassword')
                              : t('auth.userManagement.showPassword')
                          }
                          icon={showConfirmPassword ? <PiEyeSlashBold /> : <PiEyeBold />}
                          variant="ghost"
                          size="sm"
                          onClick={toggleShowConfirmPassword}
                          tabIndex={-1}
                        />
                      </Tooltip>
                    </InputRightElement>
                  </InputGroup>
                  {confirmPassword.length > 0 && !passwordsMatch && (
                    <FormErrorMessage>{t('auth.profile.passwordsDoNotMatch')}</FormErrorMessage>
                  )}
                </GridItem>
              </Grid>
            </FormControl>

            {/* Generate password button – aligned with the input column */}
            <Grid templateColumns={PASSWORD_GRID_COLUMNS} gap={4}>
              <GridItem />
              <GridItem>
                <Button size="sm" variant="ghost" onClick={handleGeneratePassword} leftIcon={<PiLightningFill />}>
                  {t('auth.userManagement.generatePassword')}
                </Button>
              </GridItem>
            </Grid>
          </Box>

          {errorMessage && (
            <Flex p={3} borderRadius="md" bg="error.500" color="white" fontSize="sm">
              <Text fontWeight="semibold">{errorMessage}</Text>
            </Flex>
          )}

          <Flex gap={3}>
            <Button variant="ghost" onClick={handleCancel}>
              {t('common.cancel')}
            </Button>
            <Button type="submit" colorScheme="invokeBlue" isLoading={isLoading} isDisabled={!isPasswordChangeValid}>
              {t('common.save')}
            </Button>
          </Flex>
        </VStack>
      </form>
    </Box>
  );
});
UserProfile.displayName = 'UserProfile';
