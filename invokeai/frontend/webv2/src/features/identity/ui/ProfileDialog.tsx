import { Dialog, HStack, Input, Portal, Stack, Text } from '@chakra-ui/react';
import { createProfileSchema, PASSWORD_RULES_HINT, type ProfileFormValues } from '@features/identity/core/schemas';
import {
  generatePassword,
  updateCurrentUser,
  type ProfileUpdateRequest,
  type UserDTO,
} from '@features/identity/data/api';
import { setSessionUser, useAuthSession } from '@features/identity/session';
import { useIdentityNotify } from '@features/identity/ui/useIdentityNotify';
import { useZodForm } from '@platform/react/useZodForm';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton, Field, FieldLabel } from '@platform/ui';
import { WandSparklesIcon } from 'lucide-react';
import { useCallback, useMemo, useState, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { AuthFormAlert } from './AuthScreen';
import { PasswordInput, PasswordStrengthMeter } from './PasswordInput';

/** Account settings: display name and password change for the signed-in user. */
export const ProfileDialog = ({ isOpen, onClose, user }: { isOpen: boolean; onClose: () => void; user: UserDTO }) => {
  const { t } = useTranslation();
  const handleOpenChange = useCallback(
    (event: Dialog.OpenChangeDetails) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Dialog.Root
      lazyMount
      open={isOpen}
      placement="center"
      scrollBehavior="inside"
      size="sm"
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
              <Stack gap="0.5">
                <Dialog.Title fontSize="md" fontWeight="700">
                  {t('auth.account')}
                </Dialog.Title>
                <Text color="fg.subtle" fontSize="xs">
                  {t('auth.signedInAs', { email: user.email })}
                </Text>
              </Stack>
            </Dialog.Header>
            <ProfileForm user={user} onClose={onClose} />
            <Dialog.CloseTrigger asChild>
              <CloseButton color="fg.muted" size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const ProfileForm = ({ onClose, user }: { onClose: () => void; user: UserDTO }) => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const notify = useIdentityNotify();
  const [isGenerating, setIsGenerating] = useState(false);
  const schema = useMemo(() => createProfileSchema(session.strictPasswordChecking), [session.strictPasswordChecking]);
  const initialValues: ProfileFormValues = useMemo(
    () => ({ confirmPassword: '', currentPassword: '', displayName: user.display_name ?? '', newPassword: '' }),
    [user.display_name]
  );
  const form = useZodForm(schema, initialValues);

  const fillGeneratedPassword = useCallback(async () => {
    setIsGenerating(true);

    try {
      const password = await generatePassword();

      form.setValue('newPassword', password);
      form.setValue('confirmPassword', password);
      notify.info(t('users.passwordGenerated'), t('auth.passwordGeneratedDescription'));
    } catch (error) {
      notify.error(t('users.couldNotGeneratePassword'), getApiErrorMessage(error, t('users.backendRejectedRequest')));
    } finally {
      setIsGenerating(false);
    }
  }, [form, notify, t]);

  const submit = useCallback(
    () =>
      form.handleSubmit(async (values) => {
        const changes: ProfileUpdateRequest = {};
        const displayName = values.displayName.trim();

        if (displayName !== (user.display_name ?? '')) {
          changes.display_name = displayName;
        }

        if (values.newPassword !== '') {
          changes.current_password = values.currentPassword;
          changes.new_password = values.newPassword;
        }

        if (Object.keys(changes).length === 0) {
          onClose();
          return;
        }

        try {
          const updated = await updateCurrentUser(changes);

          setSessionUser(updated);
        } catch (error) {
          throw new Error(getApiErrorMessage(error, t('auth.couldNotUpdateAccount')));
        }

        notify.success(t('auth.accountUpdated'));
        onClose();
      }),
    [form, notify, onClose, t, user.display_name]
  );
  const handleDisplayNameChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('displayName', event.target.value),
    [form]
  );
  const handleCurrentPasswordChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('currentPassword', event.target.value),
    [form]
  );
  const handleNewPasswordChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('newPassword', event.target.value),
    [form]
  );
  const handleConfirmPasswordChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => form.setValue('confirmPassword', event.target.value),
    [form]
  );
  const handleGeneratePassword = useCallback(() => void fillGeneratedPassword(), [fillGeneratedPassword]);
  const handleSave = useCallback(() => void submit(), [submit]);

  return (
    <>
      <Dialog.Body>
        <Stack gap="5" py="2">
          {form.formError ? <AuthFormAlert message={form.formError} tone="error" /> : null}
          <Field helpText={t('auth.profileDisplayNameHelp')} label={t('users.displayName')}>
            <Input
              autoComplete="name"
              placeholder={user.email}
              value={form.values.displayName}
              onChange={handleDisplayNameChange}
            />
          </Field>
          <Stack gap="3">
            <HStack justify="space-between">
              <FieldLabel>{t('auth.changePassword')}</FieldLabel>
              <Button loading={isGenerating} size="2xs" variant="outline" onClick={handleGeneratePassword}>
                <WandSparklesIcon />
                {t('users.generate')}
              </Button>
            </HStack>
            <Field error={form.errors.currentPassword} label={t('auth.currentPassword')}>
              <PasswordInput
                aria-invalid={form.errors.currentPassword ? true : undefined}
                autoComplete="current-password"
                value={form.values.currentPassword}
                onChange={handleCurrentPasswordChange}
              />
            </Field>
            <Field
              error={form.errors.newPassword}
              helpText={session.strictPasswordChecking ? PASSWORD_RULES_HINT : t('auth.leaveBlankKeepPassword')}
              label={t('users.newPassword')}
            >
              <Stack gap="1.5">
                <PasswordInput
                  aria-invalid={form.errors.newPassword ? true : undefined}
                  autoComplete="new-password"
                  value={form.values.newPassword}
                  onChange={handleNewPasswordChange}
                />
                <PasswordStrengthMeter password={form.values.newPassword} />
              </Stack>
            </Field>
            <Field error={form.errors.confirmPassword} label={t('auth.confirmNewPassword')}>
              <PasswordInput
                aria-invalid={form.errors.confirmPassword ? true : undefined}
                autoComplete="new-password"
                value={form.values.confirmPassword}
                onChange={handleConfirmPasswordChange}
              />
            </Field>
          </Stack>
        </Stack>
      </Dialog.Body>
      <Dialog.Footer gap="2">
        <Button size="xs" variant="ghost" onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button loading={form.isSubmitting} size="xs" variant="solid" onClick={handleSave}>
          {t('users.saveChanges')}
        </Button>
      </Dialog.Footer>
    </>
  );
};
