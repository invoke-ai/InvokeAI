import {
  Box,
  Button,
  Center,
  Checkbox,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Heading,
  Input,
  Spinner,
  Text,
  VStack,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { setCredentials } from 'features/auth/store/authSlice';
import type { ChangeEvent, FormEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { useGetSetupStatusQuery, useLoginMutation } from 'services/api/endpoints/auth';

export const LoginPage = memo(() => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(true);
  const [login, { isLoading, error }] = useLoginMutation();
  const dispatch = useAppDispatch();
  const { data: setupStatus, isLoading: isLoadingSetup } = useGetSetupStatusQuery();

  // Redirect to setup page if setup is required
  useEffect(() => {
    if (!isLoadingSetup && setupStatus?.setup_required) {
      navigate('/setup', { replace: true });
    }
  }, [setupStatus, isLoadingSetup, navigate]);

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      try {
        const result = await login({ email, password, remember_me: rememberMe }).unwrap();
        // Map the UserDTO from API to our User type
        const user = {
          user_id: result.user.user_id,
          email: result.user.email,
          display_name: result.user.display_name || null,
          is_admin: result.user.is_admin || false,
          is_active: result.user.is_active || true,
        };
        dispatch(setCredentials({ token: result.token, user }));
        // Navigate to main app after successful login
        navigate('/app', { replace: true });
      } catch {
        // Error is handled by RTK Query and displayed via error state
      }
    },
    [email, password, rememberMe, login, dispatch, navigate]
  );

  const handleEmailChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setEmail(e.target.value);
  }, []);

  const handlePasswordChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
  }, []);

  const handleRememberMeChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setRememberMe(e.target.checked);
  }, []);

  const errorMessage = error
    ? 'data' in error && typeof error.data === 'object' && error.data && 'detail' in error.data
      ? String(error.data.detail)
      : t('auth.login.loginFailed')
    : null;

  // Show loading spinner while checking setup status
  if (isLoadingSetup) {
    return (
      <Center w="100dvw" h="100dvh">
        <Spinner size="xl" />
      </Center>
    );
  }

  return (
    <Center w="100dvw" h="100dvh" bg="base.900">
      <Box w="full" maxW="400px" p={8} borderRadius="lg" bg="base.800" boxShadow="dark-lg">
        <form onSubmit={handleSubmit}>
          <VStack spacing={6} align="stretch">
            <Heading size="lg" textAlign="center">
              {t('auth.login.title')}
            </Heading>

            <FormControl isRequired isInvalid={!!errorMessage}>
              <FormLabel>{t('auth.login.email')}</FormLabel>
              <Input
                type="email"
                value={email}
                onChange={handleEmailChange}
                placeholder={t('auth.login.emailPlaceholder')}
                autoComplete="email"
                autoFocus
              />
            </FormControl>

            <FormControl isRequired isInvalid={!!errorMessage}>
              <FormLabel>{t('auth.login.password')}</FormLabel>
              <Input
                type="password"
                value={password}
                onChange={handlePasswordChange}
                placeholder={t('auth.login.passwordPlaceholder')}
                autoComplete="current-password"
              />
              {errorMessage && <FormErrorMessage>{errorMessage}</FormErrorMessage>}
            </FormControl>

            <Checkbox isChecked={rememberMe} onChange={handleRememberMeChange}>
              {t('auth.login.rememberMe')}
            </Checkbox>

            <Button
              type="submit"
              isLoading={isLoading}
              loadingText={t('auth.login.signingIn')}
              colorScheme="invokeBlue"
              size="lg"
              w="full"
            >
              {t('auth.login.signIn')}
            </Button>

            {errorMessage && (
              <Flex p={3} borderRadius="md" bg="error.500" color="white" fontSize="sm" justifyContent="center">
                <Text fontWeight="semibold">{errorMessage}</Text>
              </Flex>
            )}
          </VStack>
        </form>
      </Box>
    </Center>
  );
});

LoginPage.displayName = 'LoginPage';
