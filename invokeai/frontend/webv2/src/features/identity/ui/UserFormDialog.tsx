import { Dialog, HStack, Input, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import { createUserFormSchema, PASSWORD_RULES_HINT, type UserFormValues } from '@features/identity/core/schemas';
import {
  createUser,
  generatePassword,
  updateUser,
  type UserDTO,
  type UserUpdateRequest,
} from '@features/identity/data/api';
import { useAuthSession } from '@features/identity/session';
import { AuthFormAlert } from '@features/identity/ui/AuthScreen';
import { PasswordInput, PasswordStrengthMeter } from '@features/identity/ui/PasswordInput';
import { useIdentityNotify } from '@features/identity/ui/useIdentityNotify';
import { useZodForm } from '@platform/react/useZodForm';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton, Field } from '@platform/ui';
import { WandSparklesIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

export type UserFormTarget = { mode: 'create' } | { mode: 'edit'; user: UserDTO };

const getTargetKey = (target: UserFormTarget): string => (target.mode === 'edit' ? target.user.user_id : 'create');

const SWITCH_CHECKED_STYLES = { bg: 'accent.solid' };

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
}) => {
  const { t } = useTranslation();
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Dialog.Root
      lazyMount
      open={target !== null}
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
            {target ? (
              <>
                <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
                  <Stack gap="0.5">
                    <Dialog.Title fontSize="md" fontWeight="700">
                      {target.mode === 'create' ? t('users.addUser') : t('users.editUser')}
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
};

const UserForm = ({
  onClose,
  onSaved,
  target,
}: {
  onClose: () => void;
  onSaved: () => void;
  target: UserFormTarget;
}) => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const notify = useIdentityNotify();
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

  const fillGeneratedPassword = useCallback(async () => {
    setIsGenerating(true);

    try {
      form.setValue('password', await generatePassword());
      notify.info(t('users.passwordGenerated'), t('users.passwordGeneratedDescription'));
    } catch (error) {
      notify.error(t('users.couldNotGeneratePassword'), getApiErrorMessage(error, t('users.backendRejectedRequest')));
    } finally {
      setIsGenerating(false);
    }
  }, [form, notify, t]);

  const submit = useCallback(
    () =>
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
          throw new Error(getApiErrorMessage(error, isCreate ? t('users.couldNotCreate') : t('users.couldNotUpdate')));
        }

        notify.success(isCreate ? t('users.created') : t('users.updated'));
        onSaved();
        onClose();
      }),
    [editedUser, form, isCreate, notify, onClose, onSaved, t]
  );

  const handleEmailChange = useCallback(
    (event: { target: { value: string } }) => form.setValue('email', event.target.value),
    [form]
  );
  const handleDisplayNameChange = useCallback(
    (event: { target: { value: string } }) => form.setValue('displayName', event.target.value),
    [form]
  );
  const handlePasswordChange = useCallback(
    (event: { target: { value: string } }) => form.setValue('password', event.target.value),
    [form]
  );
  const handleGeneratedPasswordClick = useCallback(() => void fillGeneratedPassword(), [fillGeneratedPassword]);
  const handleAdminCheckedChange = useCallback(
    (event: { checked: boolean }) => form.setValue('isAdmin', event.checked),
    [form]
  );
  const handleSubmitClick = useCallback(() => void submit(), [submit]);

  return (
    <>
      <Dialog.Body>
        <Stack gap="4" py="2">
          {form.formError ? <AuthFormAlert message={form.formError} tone="error" /> : null}
          {isCreate ? (
            <Field error={form.errors.email} label={t('users.email')}>
              <Input
                aria-invalid={form.errors.email ? true : undefined}
                autoComplete="off"
                autoFocus
                placeholder="user@example.com"
                value={form.values.email}
                onChange={handleEmailChange}
              />
            </Field>
          ) : null}
          <Field helpText={t('users.displayNameHelp')} label={t('users.displayName')}>
            <Input autoComplete="off" value={form.values.displayName} onChange={handleDisplayNameChange} />
          </Field>
          <Field
            error={form.errors.password}
            helpText={
              isCreate
                ? session.strictPasswordChecking
                  ? PASSWORD_RULES_HINT
                  : undefined
                : t('users.leaveBlankPassword')
            }
            label={isCreate ? t('users.password') : t('users.newPassword')}
          >
            <Stack gap="1.5">
              <HStack gap="2">
                <PasswordInput
                  aria-invalid={form.errors.password ? true : undefined}
                  autoComplete="new-password"
                  flex="1"
                  value={form.values.password}
                  onChange={handlePasswordChange}
                />
                <Button loading={isGenerating} size="xs" variant="outline" onClick={handleGeneratedPasswordClick}>
                  <WandSparklesIcon />
                  {t('users.generate')}
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
            onCheckedChange={handleAdminCheckedChange}
          >
            <Stack gap="0.5">
              <Switch.Label color="fg" fontSize="sm" fontWeight="500" m="0">
                {t('users.administrator')}
              </Switch.Label>
              <Text color="fg.subtle" fontSize="xs">
                {isSelf ? t('users.cannotChangeSelfRole') : t('users.administratorHelp')}
              </Text>
            </Stack>
            <Switch.HiddenInput />
            <Switch.Control _checked={SWITCH_CHECKED_STYLES}>
              <Switch.Thumb />
            </Switch.Control>
          </Switch.Root>
        </Stack>
      </Dialog.Body>
      <Dialog.Footer gap="2">
        <Button size="xs" variant="ghost" onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button loading={form.isSubmitting} size="xs" variant="solid" onClick={handleSubmitClick}>
          {isCreate ? t('users.createUser') : t('users.saveChanges')}
        </Button>
      </Dialog.Footer>
    </>
  );
};
