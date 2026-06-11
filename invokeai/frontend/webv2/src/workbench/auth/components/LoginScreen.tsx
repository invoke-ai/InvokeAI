import { chakra, Checkbox, Input } from '@chakra-ui/react';
import { useNavigate } from '@tanstack/react-router';

import { AuthFormAlert, AuthScreen } from './AuthScreen';
import { PasswordInput } from './PasswordInput';
import { getApiErrorMessage } from '../../backend/http';
import { Button } from '../../components/ui/Button';
import { Field } from '../../components/ui/Field';
import { useZodForm } from '../../models/useZodForm';
import { loginSchema } from '../schemas';
import { loginWithCredentials, useAuthSession } from '../session';

/** Sign-in screen, shown only when the backend runs in multi-user mode. */
export const LoginScreen = () => {
  const session = useAuthSession();
  const navigate = useNavigate();
  const form = useZodForm(loginSchema, { email: '', password: '', rememberMe: false });

  const submit = () =>
    form.handleSubmit(async (values) => {
      try {
        await loginWithCredentials(values.email, values.password, values.rememberMe);
      } catch (error) {
        throw new Error(getApiErrorMessage(error, 'Sign-in failed. Check your email and password.'));
      }

      await navigate({ to: '/' });
    });

  return (
    <AuthScreen subtitle="Sign in to your workspace to continue." title="Welcome to Invoke">
      <chakra.form
        display="flex"
        flexDirection="column"
        gap="4"
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
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
            onChange={(event) => form.setValue('email', event.target.value)}
          />
        </Field>
        <Field error={form.errors.password} label="Password">
          <PasswordInput
            aria-invalid={form.errors.password ? true : undefined}
            autoComplete="current-password"
            placeholder="Your password"
            value={form.values.password}
            onChange={(event) => form.setValue('password', event.target.value)}
          />
        </Field>
        <Checkbox.Root
          checked={form.values.rememberMe}
          size="sm"
          onCheckedChange={(event) => form.setValue('rememberMe', event.checked === true)}
        >
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
