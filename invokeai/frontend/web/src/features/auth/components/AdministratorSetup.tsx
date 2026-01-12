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
  Input,
  Text,
  VStack,
} from '@invoke-ai/ui-library';
import type { ChangeEvent, FormEvent } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSetupMutation } from 'services/api/endpoints/auth';

const validatePasswordStrength = (
  password: string,
  t: (key: string) => string
): { isValid: boolean; message: string } => {
  if (password.length < 8) {
    return { isValid: false, message: t('auth.setup.passwordTooShort') };
  }

  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);

  if (!hasUpper || !hasLower || !hasDigit) {
    return {
      isValid: false,
      message: t('auth.setup.passwordMissingRequirements'),
    };
  }

  return { isValid: true, message: '' };
};

export const AdministratorSetup = memo(() => {
  const { t } = useTranslation();
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [setup, { isLoading, error }] = useSetupMutation();

  const passwordValidation = validatePasswordStrength(password, t);
  const passwordsMatch = password === confirmPassword;

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();

      if (!passwordValidation.isValid) {
        return;
      }

      if (!passwordsMatch) {
        return;
      }

      try {
        const result = await setup({ email, display_name: displayName, password }).unwrap();
        if (result.success) {
          // Auto-login after setup - need to call login API
          // For now, just redirect to login page
          window.location.href = '/login';
        }
      } catch {
        // Error is handled by RTK Query and displayed via error state
      }
    },
    [email, displayName, password, passwordValidation.isValid, passwordsMatch, setup]
  );

  const handleEmailChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setEmail(e.target.value);
  }, []);

  const handleDisplayNameChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setDisplayName(e.target.value);
  }, []);

  const handlePasswordChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
  }, []);

  const handleConfirmPasswordChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setConfirmPassword(e.target.value);
  }, []);

  const errorMessage = error
    ? 'data' in error && typeof error.data === 'object' && error.data && 'detail' in error.data
      ? String(error.data.detail)
      : t('auth.setup.setupFailed')
    : null;

  return (
    <Center w="100dvw" h="100dvh" bg="base.900">
      <Box w="full" maxW="600px" p={8} borderRadius="lg" bg="base.800" boxShadow="dark-lg">
        <form onSubmit={handleSubmit}>
          <VStack spacing={6} align="stretch">
            <VStack spacing={2}>
              <Heading size="lg" textAlign="center">
                {t('auth.setup.title')}
              </Heading>
              <Text fontSize="sm" color="base.400" textAlign="center">
                {t('auth.setup.subtitle')}
              </Text>
            </VStack>

            <FormControl isRequired>
              <Grid templateColumns="140px 1fr" gap={4} alignItems="start">
                <GridItem>
                  <FormLabel textAlign="right" pt={2} mb={0}>
                    {t('auth.setup.email')}
                  </FormLabel>
                </GridItem>
                <GridItem>
                  <Input
                    type="email"
                    value={email}
                    onChange={handleEmailChange}
                    placeholder={t('auth.setup.emailPlaceholder')}
                    autoComplete="email"
                    autoFocus
                  />
                  <FormHelperText mt={1}>{t('auth.setup.emailHelper')}</FormHelperText>
                </GridItem>
              </Grid>
            </FormControl>

            <FormControl isRequired>
              <Grid templateColumns="140px 1fr" gap={4} alignItems="start">
                <GridItem>
                  <FormLabel textAlign="right" pt={2} mb={0}>
                    {t('auth.setup.displayName')}
                  </FormLabel>
                </GridItem>
                <GridItem>
                  <Input
                    type="text"
                    value={displayName}
                    onChange={handleDisplayNameChange}
                    placeholder={t('auth.setup.displayNamePlaceholder')}
                  />
                  <FormHelperText mt={1}>{t('auth.setup.displayNameHelper')}</FormHelperText>
                </GridItem>
              </Grid>
            </FormControl>

            <FormControl isRequired isInvalid={password.length > 0 && !passwordValidation.isValid}>
              <Grid templateColumns="140px 1fr" gap={4} alignItems="start">
                <GridItem>
                  <FormLabel textAlign="right" pt={2} mb={0}>
                    {t('auth.setup.password')}
                  </FormLabel>
                </GridItem>
                <GridItem>
                  <Input
                    type="password"
                    value={password}
                    onChange={handlePasswordChange}
                    placeholder={t('auth.setup.passwordPlaceholder')}
                    autoComplete="new-password"
                  />
                  {password.length > 0 && !passwordValidation.isValid && (
                    <FormErrorMessage>{passwordValidation.message}</FormErrorMessage>
                  )}
                  {password.length === 0 && <FormHelperText mt={1}>{t('auth.setup.passwordHelper')}</FormHelperText>}
                </GridItem>
              </Grid>
            </FormControl>

            <FormControl isRequired isInvalid={confirmPassword.length > 0 && !passwordsMatch}>
              <Grid templateColumns="140px 1fr" gap={4} alignItems="start">
                <GridItem>
                  <FormLabel textAlign="right" pt={2} mb={0}>
                    {t('auth.setup.confirmPassword')}
                  </FormLabel>
                </GridItem>
                <GridItem>
                  <Input
                    type="password"
                    value={confirmPassword}
                    onChange={handleConfirmPasswordChange}
                    placeholder={t('auth.setup.confirmPasswordPlaceholder')}
                    autoComplete="new-password"
                  />
                  {confirmPassword.length > 0 && !passwordsMatch && (
                    <FormErrorMessage>{t('auth.setup.passwordsDoNotMatch')}</FormErrorMessage>
                  )}
                </GridItem>
              </Grid>
            </FormControl>

            <Button
              type="submit"
              isLoading={isLoading}
              loadingText={t('auth.setup.creatingAccount')}
              colorScheme="invokeBlue"
              size="lg"
              w="full"
              isDisabled={!passwordValidation.isValid || !passwordsMatch}
            >
              {t('auth.setup.createAccount')}
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

AdministratorSetup.displayName = 'AdministratorSetup';
