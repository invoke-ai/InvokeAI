import { Dialog, HStack, Input, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import { createUser, generatePassword, updateUser, type UserDTO, type UserUpdateRequest } from '@workbench/auth/api';
import { AuthFormAlert } from '@workbench/auth/components/AuthScreen';
import { PasswordInput, PasswordStrengthMeter } from '@workbench/auth/components/PasswordInput';
import { createUserFormSchema, PASSWORD_RULES_HINT, type UserFormValues } from '@workbench/auth/schemas';
import { useAuthSession } from '@workbench/auth/session';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button, CloseButton, Field } from '@workbench/components/ui';
import { useZodForm } from '@workbench/models/useZodForm';
import { useNotify } from '@workbench/useNotify';
import { WandSparklesIcon } from 'lucide-react';
import { useMemo, useState } from 'react';

export type UserFormTarget = { mode: 'create' } | { mode: 'edit'; user: UserDTO };

const getTargetKey = (target: UserFormTarget): string => (target.mode === 'edit' ? target.user.user_id : 'create');

/**
 * Create/edit user form for the users widget. The dialog root stays mounted
 * and is driven by `target` (null = closed); the inner form is keyed by the
 * target so each open starts from a fresh state.
 */
export const UserFormDialog = ({
  onClose,
  onSaved,
  target,
}: {
  onClose: () => void;
  onSaved: () => void;
  target: UserFormTarget | null;
}) => (
  <Dialog.Root
    lazyMount
    open={target !== null}
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
          {target ? (
            <>
              <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
                <Stack gap="0.5">
                  <Dialog.Title fontSize="md" fontWeight="700">
                    {target.mode === 'create' ? 'Add user' : 'Edit user'}
                  </Dialog.Title>
                  {target.mode === 'edit' ? (
                    <Text color="fg.subtle" fontSize="xs">
                      {target.user.email}
                    </Text>
                  ) : null}
                </Stack>
              </Dialog.Header>
              <UserForm key={getTargetKey(target)} target={target} onClose={onClose} onSaved={onSaved} />
            </>
          ) : null}
          <Dialog.CloseTrigger asChild>
            <CloseButton color="fg.muted" size="sm" />
          </Dialog.CloseTrigger>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  </Dialog.Root>
);

const UserForm = ({
  onClose,
  onSaved,
  target,
}: {
  onClose: () => void;
  onSaved: () => void;
  target: UserFormTarget;
}) => {
  const session = useAuthSession();
  const notify = useNotify();
  const [isGenerating, setIsGenerating] = useState(false);
  const isCreate = target.mode === 'create';
  const editedUser = target.mode === 'edit' ? target.user : null;
  const isSelf = editedUser !== null && editedUser.user_id === session.user?.user_id;

  const schema = useMemo(
    () => createUserFormSchema(session.strictPasswordChecking, isCreate),
    [isCreate, session.strictPasswordChecking]
  );
  const initialValues: UserFormValues = useMemo(
    () => ({
      displayName: editedUser?.display_name ?? '',
      email: '',
      isAdmin: editedUser?.is_admin ?? false,
      password: '',
    }),
    [editedUser]
  );
  const form = useZodForm(schema, initialValues);

  const fillGeneratedPassword = async () => {
    setIsGenerating(true);

    try {
      form.setValue('password', await generatePassword());
      notify.info('Password generated', 'Reveal it with the eye icon and share it with the user securely.');
    } catch (error) {
      notify.error('Could not generate a password', getApiErrorMessage(error, 'The backend rejected the request.'));
    } finally {
      setIsGenerating(false);
    }
  };

  const submit = () =>
    form.handleSubmit(async (values) => {
      try {
        if (isCreate) {
          await createUser({
            display_name: values.displayName.trim() || null,
            email: values.email,
            is_admin: values.isAdmin,
            password: values.password,
          });
        } else if (editedUser) {
          const changes: UserUpdateRequest = {};
          const displayName = values.displayName.trim();

          if (displayName !== (editedUser.display_name ?? '')) {
            changes.display_name = displayName;
          }

          if (values.password !== '') {
            changes.password = values.password;
          }

          if (values.isAdmin !== editedUser.is_admin) {
            changes.is_admin = values.isAdmin;
          }

          if (Object.keys(changes).length === 0) {
            onClose();
            return;
          }

          await updateUser(editedUser.user_id, changes);
        }
      } catch (error) {
        throw new Error(
          getApiErrorMessage(error, isCreate ? 'Could not create the user.' : 'Could not update the user.')
        );
      }

      notify.success(isCreate ? 'User created' : 'User updated');
      onSaved();
      onClose();
    });

  return (
    <>
      <Dialog.Body>
        <Stack gap="4" py="2">
          {form.formError ? <AuthFormAlert message={form.formError} tone="error" /> : null}
          {isCreate ? (
            <Field error={form.errors.email} label="Email">
              <Input
                aria-invalid={form.errors.email ? true : undefined}
                autoComplete="off"
                autoFocus
                placeholder="user@example.com"
                value={form.values.email}
                onChange={(event) => form.setValue('email', event.target.value)}
              />
            </Field>
          ) : null}
          <Field helpText="Optional — shown instead of the email." label="Display name">
            <Input
              autoComplete="off"
              value={form.values.displayName}
              onChange={(event) => form.setValue('displayName', event.target.value)}
            />
          </Field>
          <Field
            error={form.errors.password}
            helpText={
              isCreate
                ? session.strictPasswordChecking
                  ? PASSWORD_RULES_HINT
                  : undefined
                : 'Leave blank to keep the current password.'
            }
            label={isCreate ? 'Password' : 'New password'}
          >
            <Stack gap="1.5">
              <HStack gap="2">
                <PasswordInput
                  aria-invalid={form.errors.password ? true : undefined}
                  autoComplete="new-password"
                  flex="1"
                  value={form.values.password}
                  onChange={(event) => form.setValue('password', event.target.value)}
                />
                <Button loading={isGenerating} size="xs" variant="outline" onClick={() => void fillGeneratedPassword()}>
                  <WandSparklesIcon />
                  Generate
                </Button>
              </HStack>
              <PasswordStrengthMeter password={form.values.password} />
            </Stack>
          </Field>
          <Switch.Root
            alignItems="center"
            checked={form.values.isAdmin}
            disabled={isSelf}
            display="flex"
            justifyContent="space-between"
            w="full"
            onCheckedChange={(event) => form.setValue('isAdmin', event.checked)}
          >
            <Stack gap="0.5">
              <Switch.Label color="fg" fontSize="sm" fontWeight="500" m="0">
                Administrator
              </Switch.Label>
              <Text color="fg.subtle" fontSize="xs">
                {isSelf ? 'You cannot change your own role.' : 'Can manage users and all workspace settings.'}
              </Text>
            </Stack>
            <Switch.HiddenInput />
            <Switch.Control _checked={{ bg: 'accent.solid' }}>
              <Switch.Thumb />
            </Switch.Control>
          </Switch.Root>
        </Stack>
      </Dialog.Body>
      <Dialog.Footer gap="2">
        <Button size="xs" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
        <Button loading={form.isSubmitting} size="xs" variant="solid" onClick={() => void submit()}>
          {isCreate ? 'Create user' : 'Save changes'}
        </Button>
      </Dialog.Footer>
    </>
  );
};
