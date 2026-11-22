import { Alert, chakra, Icon, Input, Link, Stack } from '@chakra-ui/react';
import { createSetupSchema, PASSWORD_RULES_HINT, type SetupFormValues } from '@features/identity/core/schemas';
import { completeAdminSetup, isLoginAttemptSupersededError, useAuthSession } from '@features/identity/session';
import { useZodForm } from '@platform/react/useZodForm';
import { DOCS_URL } from '@platform/runtime/appMetadata';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, Field } from '@platform/ui';
import { Navigate } from '@tanstack/react-router';
import { ExternalLinkIcon } from 'lucide-react';
import { useCallback, useId, useMemo, type ChangeEvent, type FormEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { AuthFormAlert, AuthScreen } from './AuthScreen';
import { PasswordInput, PasswordStrengthMeter } from './PasswordInput';

const INITIAL_VALUES: SetupFormValues = { confirmPassword: '', displayName: '', email: '', password: '' };
const EARLIER_WORK_GUIDE_URL = `${DOCS_URL}configuration/multi-user-mode/admin-guide/#if-you-have-already-upgraded`;

export const SetupScreen = () => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const schema = useMemo(() => createSetupSchema(session.strictPasswordChecking), [session.strictPasswordChecking]);
  const form = useZodForm(schema, INITIAL_VALUES);
  const earlierWorkDescriptionId = useId();

  const submit = useCallback(
    () =>
      form.handleSubmit(async (values) => {
        try {
          await completeAdminSetup(values.email, values.displayName.trim() || null, values.password);
        } catch (error) {
          if (isLoginAttemptSupersededError(error)) {
            return;
          }

          throw new Error(getApiErrorMessage(error, t('auth.couldNotCreateAdmin')));
        }
      }),
    [form, t]
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

  // Leave once signed in, whether by this form or by following another tab that completed setup.
  if (session.multiuserEnabled && session.user !== null) {
    return <Navigate replace to="/" />;
  }

  return (
    <AuthScreen footer={t('auth.setupFooter')} subtitle={t('auth.setupSubtitle')} title={t('auth.setupTitle')}>
      <chakra.form display="flex" flexDirection="column" gap="4" onSubmit={handleSubmit}>
        {form.formError ? <AuthFormAlert message={form.formError} tone="error" /> : null}
        <Field error={form.errors.email} label={t('users.email')}>
          <Input
            aria-invalid={form.errors.email ? true : undefined}
            autoComplete="email"
            autoFocus
            placeholder="admin@example.com"
            size="lg"
            value={form.values.email}
            onChange={handleEmailChange}
          />
        </Field>
        <Field helpText={t('auth.displayNameHelp')} label={t('users.displayName')}>
          <Input
            autoComplete="name"
            placeholder={t('auth.administrator')}
            size="lg"
            value={form.values.displayName}
            onChange={handleDisplayNameChange}
          />
        </Field>
        <Field
          error={form.errors.password}
          helpText={session.strictPasswordChecking ? PASSWORD_RULES_HINT : undefined}
          label={t('auth.password')}
        >
          <Stack gap="1.5">
            <PasswordInput
              aria-invalid={form.errors.password ? true : undefined}
              autoComplete="new-password"
              size="lg"
              value={form.values.password}
              onChange={handlePasswordChange}
            />
            <PasswordStrengthMeter password={form.values.password} />
          </Stack>
        </Field>
        <Field error={form.errors.confirmPassword} label={t('auth.confirmPassword')}>
          <PasswordInput
            aria-invalid={form.errors.confirmPassword ? true : undefined}
            autoComplete="new-password"
            size="lg"
            value={form.values.confirmPassword}
            onChange={handleConfirmPasswordChange}
          />
        </Field>
        {/* Above the submit, which it describes, so keyboard and screen-reader users hear it on reaching the button. */}
        {session.hasSingleUserWork ? <EarlierWorkNotice descriptionId={earlierWorkDescriptionId} /> : null}
        <Button
          aria-describedby={session.hasSingleUserWork ? earlierWorkDescriptionId : undefined}
          loading={form.isSubmitting}
          size="lg"
          type="submit"
          variant="solid"
        >
          {t('auth.createAdminAccount')}
        </Button>
      </chakra.form>
    </AuthScreen>
  );
};

const EarlierWorkNotice = ({ descriptionId }: { descriptionId: string }) => {
  const { t } = useTranslation();

  return (
    <Alert.Root borderRadius="md" role="note" status="info" variant="surface">
      <Alert.Indicator />
      <Alert.Content gap="1.5">
        <Alert.Title fontSize="md" fontWeight="600">
          {t('auth.setupEarlierWorkTitle')}
        </Alert.Title>
        <Alert.Description fontSize="md" id={descriptionId}>
          {t('auth.setupEarlierWorkDescription')}
        </Alert.Description>
        {/* Inline, so a wrapped label keeps the icon after its last word. */}
        <Link
          alignSelf="flex-start"
          display="inline"
          fontSize="md"
          fontWeight="600"
          href={EARLIER_WORK_GUIDE_URL}
          rel="noreferrer"
          target="_blank"
          variant="underline"
        >
          {t('auth.setupEarlierWorkGuide')}
          <Icon as={ExternalLinkIcon} boxSize="3" ms="1" verticalAlign="-1px" />
        </Link>
      </Alert.Content>
    </Alert.Root>
  );
};
