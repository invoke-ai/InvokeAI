import { chakra, Checkbox, Input } from '@chakra-ui/react';
import { loginSchema } from '@features/identity/core/schemas';
import { loginWithCredentials, useAuthSession } from '@features/identity/session';
import { useZodForm } from '@platform/react/useZodForm';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, Field } from '@platform/ui';
import { useNavigate } from '@tanstack/react-router';
import { useCallback, type ChangeEvent, type FormEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { AuthFormAlert, AuthScreen } from './AuthScreen';
import { PasswordInput } from './PasswordInput';

/** Sign-in screen, shown only when the backend runs in multi-user mode. */
export const LoginScreen = () => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const navigate = useNavigate();
  const form = useZodForm(loginSchema, { email: '', password: '', rememberMe: false });

  const submit = useCallback(
    () =>
      form.handleSubmit(async (values) => {
        try {
          await loginWithCredentials(values.email, values.password, values.rememberMe);
        } catch (error) {
          throw new Error(getApiErrorMessage(error, t('auth.signInFailed')));
        }

        await navigate({ to: '/' });
      }),
    [form, navigate, t]
  );

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      void submit();
    },
    [submit]
  );

  const handleEmailChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('email', event.target.value),
    [form]
  );

  const handlePasswordChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('password', event.target.value),
    [form]
  );

  const handleRememberMeChange = useCallback(
    (event: Checkbox.CheckedChangeDetails) => form.setValue('rememberMe', event.checked === true),
    [form]
  );

  return (
    <AuthScreen subtitle={t('auth.signInSubtitle')} title={t('auth.welcomeTitle')}>
      <chakra.form display="flex" flexDirection="column" gap="4" onSubmit={handleSubmit}>
        {session.sessionExpired ? <AuthFormAlert message={t('auth.sessionExpired')} tone="warning" /> : null}
        {form.formError ? <AuthFormAlert message={form.formError} tone="error" /> : null}
        <Field error={form.errors.email} label={t('users.email')}>
          <Input
            aria-invalid={form.errors.email ? true : undefined}
            autoComplete="email"
            autoFocus
            placeholder="you@example.com"
            value={form.values.email}
            onChange={handleEmailChange}
          />
        </Field>
        <Field error={form.errors.password} label={t('auth.password')}>
          <PasswordInput
            aria-invalid={form.errors.password ? true : undefined}
            autoComplete="current-password"
            placeholder={t('auth.yourPassword')}
            value={form.values.password}
            onChange={handlePasswordChange}
          />
        </Field>
        <Checkbox.Root checked={form.values.rememberMe} size="sm" onCheckedChange={handleRememberMeChange}>
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label color="fg.muted" fontWeight="400">
            {t('auth.keepSignedIn')}
          </Checkbox.Label>
        </Checkbox.Root>
        <Button loading={form.isSubmitting} size="sm" type="submit" variant="solid">
          {t('auth.signIn')}
        </Button>
      </chakra.form>
    </AuthScreen>
  );
};
