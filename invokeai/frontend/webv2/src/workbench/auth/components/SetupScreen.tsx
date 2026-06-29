import { chakra, Input, Stack } from '@chakra-ui/react';
import { useNavigate } from '@tanstack/react-router';
import { createSetupSchema, PASSWORD_RULES_HINT, type SetupFormValues } from '@workbench/auth/schemas';
import { completeAdminSetup, useAuthSession } from '@workbench/auth/session';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button, Field } from '@workbench/components/ui';
import { useZodForm } from '@workbench/models/useZodForm';
import { useCallback, useMemo, type ChangeEvent, type FormEvent } from 'react';

import { AuthFormAlert, AuthScreen } from './AuthScreen';
import { PasswordInput, PasswordStrengthMeter } from './PasswordInput';

const INITIAL_VALUES: SetupFormValues = { confirmPassword: '', displayName: '', email: '', password: '' };

/**
 * First-run screen for multi-user mode: creates the administrator account and
 * signs straight into it.
 */
export const SetupScreen = () => {
  const session = useAuthSession();
  const navigate = useNavigate();
  const schema = useMemo(() => createSetupSchema(session.strictPasswordChecking), [session.strictPasswordChecking]);
  const form = useZodForm(schema, INITIAL_VALUES);

  const submit = useCallback(
    () =>
      form.handleSubmit(async (values) => {
        try {
          await completeAdminSetup(values.email, values.displayName.trim() || null, values.password);
        } catch (error) {
          throw new Error(getApiErrorMessage(error, 'Could not create the administrator account.'));
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
  const handleDisplayNameChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('displayName', event.target.value),
    [form]
  );
  const handlePasswordChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('password', event.target.value),
    [form]
  );
  const handleConfirmPasswordChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('confirmPassword', event.target.value),
    [form]
  );

  return (
    <AuthScreen
      footer="You can add more users later from the user menu."
      subtitle="Create the administrator account to get started."
      title="Set up Invoke"
    >
      <chakra.form display="flex" flexDirection="column" gap="4" onSubmit={handleSubmit}>
        {form.formError ? <AuthFormAlert message={form.formError} tone="error" /> : null}
        <Field error={form.errors.email} label="Email">
          <Input
            aria-invalid={form.errors.email ? true : undefined}
            autoComplete="email"
            autoFocus
            placeholder="admin@example.com"
            value={form.values.email}
            onChange={handleEmailChange}
          />
        </Field>
        <Field helpText="Optional — shown instead of your email." label="Display name">
          <Input
            autoComplete="name"
            placeholder="Administrator"
            value={form.values.displayName}
            onChange={handleDisplayNameChange}
          />
        </Field>
        <Field
          error={form.errors.password}
          helpText={session.strictPasswordChecking ? PASSWORD_RULES_HINT : undefined}
          label="Password"
        >
          <Stack gap="1.5">
            <PasswordInput
              aria-invalid={form.errors.password ? true : undefined}
              autoComplete="new-password"
              value={form.values.password}
              onChange={handlePasswordChange}
            />
            <PasswordStrengthMeter password={form.values.password} />
          </Stack>
        </Field>
        <Field error={form.errors.confirmPassword} label="Confirm password">
          <PasswordInput
            aria-invalid={form.errors.confirmPassword ? true : undefined}
            autoComplete="new-password"
            value={form.values.confirmPassword}
            onChange={handleConfirmPasswordChange}
          />
        </Field>
        <Button loading={form.isSubmitting} size="sm" type="submit" variant="solid">
          Create administrator account
        </Button>
      </chakra.form>
    </AuthScreen>
  );
};
