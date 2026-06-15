import { Dialog, HStack, Input, Portal, Stack, Text } from '@chakra-ui/react';
import { useMemo, useState } from 'react';
import { WandSparklesIcon } from 'lucide-react';

import { AuthFormAlert } from './AuthScreen';
import { PasswordInput, PasswordStrengthMeter } from './PasswordInput';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button, CloseButton } from '@workbench/components/ui/Button';
import { Field, FieldLabel } from '@workbench/components/ui/Field';
import { useZodForm } from '@workbench/models/useZodForm';
import { useNotify } from '@workbench/useNotify';
import { generatePassword, updateCurrentUser, type ProfileUpdateRequest, type UserDTO } from '@workbench/auth/api';
import { createProfileSchema, PASSWORD_RULES_HINT, type ProfileFormValues } from '@workbench/auth/schemas';
import { setSessionUser, useAuthSession } from '@workbench/auth/session';

/** Account settings: display name and password change for the signed-in user. */
export const ProfileDialog = ({ isOpen, onClose, user }: { isOpen: boolean; onClose: () => void; user: UserDTO }) => (
  <Dialog.Root
    lazyMount
    open={isOpen}
    placement="center"
    scrollBehavior="inside"
    size="sm"
    unmountOnExit
    onOpenChange={(event) => {
      if (!event.open) {
        onClose();
      }
    }}
  >
    <Portal>
      <Dialog.Backdrop />
      <Dialog.Positioner>
        <Dialog.Content>
          <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
            <Stack gap="0.5">
              <Dialog.Title fontSize="md" fontWeight="700">
                Account
              </Dialog.Title>
              <Text color="fg.subtle" fontSize="xs">
                Signed in as {user.email}
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

const ProfileForm = ({ onClose, user }: { onClose: () => void; user: UserDTO }) => {
  const session = useAuthSession();
  const notify = useNotify();
  const [isGenerating, setIsGenerating] = useState(false);
  const schema = useMemo(() => createProfileSchema(session.strictPasswordChecking), [session.strictPasswordChecking]);
  const initialValues: ProfileFormValues = useMemo(
    () => ({ confirmPassword: '', currentPassword: '', displayName: user.display_name ?? '', newPassword: '' }),
    [user.display_name]
  );
  const form = useZodForm(schema, initialValues);

  const fillGeneratedPassword = async () => {
    setIsGenerating(true);

    try {
      const password = await generatePassword();

      form.setValue('newPassword', password);
      form.setValue('confirmPassword', password);
      notify.info('Password generated', 'Reveal it with the eye icon and store it somewhere safe.');
    } catch (error) {
      notify.error('Could not generate a password', getApiErrorMessage(error, 'The backend rejected the request.'));
    } finally {
      setIsGenerating(false);
    }
  };

  const submit = () =>
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
        throw new Error(getApiErrorMessage(error, 'Could not update your account.'));
      }

      notify.success('Account updated');
      onClose();
    });

  return (
    <>
      <Dialog.Body>
        <Stack gap="5" py="2">
          {form.formError ? <AuthFormAlert message={form.formError} tone="error" /> : null}
          <Field helpText="Shown instead of your email across the workspace." label="Display name">
            <Input
              autoComplete="name"
              placeholder={user.email}
              value={form.values.displayName}
              onChange={(event) => form.setValue('displayName', event.target.value)}
            />
          </Field>
          <Stack gap="3">
            <HStack justify="space-between">
              <FieldLabel>Change password</FieldLabel>
              <Button loading={isGenerating} size="2xs" variant="outline" onClick={() => void fillGeneratedPassword()}>
                <WandSparklesIcon />
                Generate
              </Button>
            </HStack>
            <Field error={form.errors.currentPassword} label="Current password">
              <PasswordInput
                aria-invalid={form.errors.currentPassword ? true : undefined}
                autoComplete="current-password"
                value={form.values.currentPassword}
                onChange={(event) => form.setValue('currentPassword', event.target.value)}
              />
            </Field>
            <Field
              error={form.errors.newPassword}
              helpText={session.strictPasswordChecking ? PASSWORD_RULES_HINT : 'Leave blank to keep your password.'}
              label="New password"
            >
              <Stack gap="1.5">
                <PasswordInput
                  aria-invalid={form.errors.newPassword ? true : undefined}
                  autoComplete="new-password"
                  value={form.values.newPassword}
                  onChange={(event) => form.setValue('newPassword', event.target.value)}
                />
                <PasswordStrengthMeter password={form.values.newPassword} />
              </Stack>
            </Field>
            <Field error={form.errors.confirmPassword} label="Confirm new password">
              <PasswordInput
                aria-invalid={form.errors.confirmPassword ? true : undefined}
                autoComplete="new-password"
                value={form.values.confirmPassword}
                onChange={(event) => form.setValue('confirmPassword', event.target.value)}
              />
            </Field>
          </Stack>
        </Stack>
      </Dialog.Body>
      <Dialog.Footer gap="2">
        <Button size="xs" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
        <Button loading={form.isSubmitting} size="xs" variant="solid" onClick={() => void submit()}>
          Save changes
        </Button>
      </Dialog.Footer>
    </>
  );
};
