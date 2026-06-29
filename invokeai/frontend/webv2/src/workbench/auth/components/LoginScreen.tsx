import { chakra, Checkbox, Input } from '@chakra-ui/react';
import { useNavigate } from '@tanstack/react-router';
import { loginSchema } from '@workbench/auth/schemas';
import { loginWithCredentials, useAuthSession } from '@workbench/auth/session';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button, Field } from '@workbench/components/ui';
import { useZodForm } from '@workbench/models/useZodForm';
import { useCallback, type ChangeEvent, type FormEvent } from 'react';

import { AuthFormAlert, AuthScreen } from './AuthScreen';
import { PasswordInput } from './PasswordInput';

/** Sign-in screen, shown only when the backend runs in multi-user mode. */
export const LoginScreen = () => {
  const session = useAuthSession();
  const navigate = useNavigate();
  const form = useZodForm(loginSchema, { email: '', password: '', rememberMe: false });

  const submit = useCallback(
    () =>
      form.handleSubmit(async (values) => {
        try {
          await loginWithCredentials(values.email, values.password, values.rememberMe);
        } catch (error) {
          throw new Error(getApiErrorMessage(error, 'Sign-in failed. Check your email and password.'));
        }

        await navigate({ to: '/' });
      }),
    [form, navigate]
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
    <AuthScreen subtitle="Sign in to your workspace to continue." title="Welcome to Invoke">
      <chakra.form display="flex" flexDirection="column" gap="4" onSubmit={handleSubmit}>
        {session.sessionExpired ? (
          <AuthFormAlert message="Your session expired. Sign in again to continue." tone="warning" />
        ) : null}
        {form.formError ? <AuthFormAlert message={form.formError} tone="error" /> : null}
        <Field error={form.errors.email} label="Email">
          <Input
            aria-invalid={form.errors.email ? true : undefined}
            autoComplete="email"
            autoFocus
            placeholder="you@example.com"
            value={form.values.email}
            onChange={handleEmailChange}
          />
        </Field>
        <Field error={form.errors.password} label="Password">
          <PasswordInput
            aria-invalid={form.errors.password ? true : undefined}
            autoComplete="current-password"
            placeholder="Your password"
            value={form.values.password}
            onChange={handlePasswordChange}
          />
        </Field>
        <Checkbox.Root checked={form.values.rememberMe} size="sm" onCheckedChange={handleRememberMeChange}>
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label color="fg.muted" fontWeight="400">
            Keep me signed in for a week
          </Checkbox.Label>
        </Checkbox.Root>
        <Button loading={form.isSubmitting} size="sm" type="submit" variant="solid">
          Sign in
        </Button>
      </chakra.form>
    </AuthScreen>
  );
};
