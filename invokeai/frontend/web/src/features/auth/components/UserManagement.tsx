import {
  Badge,
  Box,
  Button,
  Center,
  Checkbox,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Grid,
  GridItem,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Spinner,
  Switch,
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tooltip,
  Tr,
  useDisclosure,
  VStack,
} from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import type { ChangeEvent, FormEvent } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowLeftBold,
  PiEyeBold,
  PiEyeSlashBold,
  PiLightningFill,
  PiPencilBold,
  PiPlusBold,
  PiTrashBold,
} from 'react-icons/pi';
import { useNavigate } from 'react-router-dom';
import type { UserDTO } from 'services/api/endpoints/auth';
import {
  useCreateUserMutation,
  useDeleteUserMutation,
  useLazyGeneratePasswordQuery,
  useListUsersQuery,
  useUpdateUserMutation,
} from 'services/api/endpoints/auth';

const validatePasswordStrength = (
  password: string,
  t: (key: string) => string
): { isValid: boolean; message: string } => {
  if (password.length === 0) {
    return { isValid: true, message: '' };
  }
  if (password.length < 8) {
    return { isValid: false, message: t('auth.setup.passwordTooShort') };
  }
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);
  if (!hasUpper || !hasLower || !hasDigit) {
    return { isValid: false, message: t('auth.setup.passwordMissingRequirements') };
  }
  return { isValid: true, message: '' };
};

const FORM_GRID_COLUMNS = '120px 1fr';

// ---------------------------------------------------------------------------
// Create / Edit user modal
// ---------------------------------------------------------------------------

type UserFormModalProps = {
  isOpen: boolean;
  onClose: () => void;
  /** When provided, the modal operates in "edit" mode for the given user */
  editUser?: UserDTO | null;
};

const UserFormModal = memo(({ isOpen, onClose, editUser }: UserFormModalProps) => {
  const { t } = useTranslation();
  const isEdit = !!editUser;

  const [email, setEmail] = useState(editUser?.email ?? '');
  const [displayName, setDisplayName] = useState(editUser?.display_name ?? '');
  const [password, setPassword] = useState('');
  const [isAdmin, setIsAdmin] = useState(editUser?.is_admin ?? false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [createUser, { isLoading: isCreating }] = useCreateUserMutation();
  const [updateUser, { isLoading: isUpdating }] = useUpdateUserMutation();
  const [triggerGeneratePassword] = useLazyGeneratePasswordQuery();

  const isLoading = isCreating || isUpdating;
  const passwordValidation = validatePasswordStrength(password, t);

  const handleGeneratePassword = useCallback(async () => {
    try {
      const result = await triggerGeneratePassword().unwrap();
      setPassword(result.password);
      setShowPassword(true);
    } catch {
      // ignore
    }
  }, [triggerGeneratePassword]);

  const toggleShowPassword = useCallback(() => {
    setShowPassword((v) => !v);
  }, []);

  const handleEmailChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setEmail(e.target.value);
  }, []);

  const handleDisplayNameChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setDisplayName(e.target.value);
  }, []);

  const handlePasswordChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
  }, []);

  const handleIsAdminChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setIsAdmin(e.target.checked);
  }, []);

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      setError(null);

      if (!isEdit && (!password || !passwordValidation.isValid)) {
        return;
      }
      if (isEdit && password && !passwordValidation.isValid) {
        return;
      }

      try {
        if (isEdit && editUser) {
          const updateData: Parameters<typeof updateUser>[0]['data'] = {
            display_name: displayName || null,
            is_admin: isAdmin,
          };
          if (password) {
            updateData.password = password;
          }
          await updateUser({
            userId: editUser.user_id,
            data: updateData,
          }).unwrap();
        } else {
          await createUser({
            email,
            display_name: displayName || null,
            password,
            is_admin: isAdmin,
          }).unwrap();
        }
        onClose();
      } catch (err) {
        const detail =
          err && typeof err === 'object' && 'data' in err && typeof (err as { data: unknown }).data === 'object'
            ? ((err as { data: { detail?: string } }).data?.detail ?? t('auth.userManagement.saveFailed'))
            : t('auth.userManagement.saveFailed');
        setError(detail);
      }
    },
    [
      isEdit,
      editUser,
      email,
      displayName,
      password,
      isAdmin,
      passwordValidation.isValid,
      createUser,
      updateUser,
      onClose,
      t,
    ]
  );

  // Reset local state when modal closes
  const handleClose = useCallback(() => {
    setEmail(editUser?.email ?? '');
    setDisplayName(editUser?.display_name ?? '');
    setPassword('');
    setIsAdmin(editUser?.is_admin ?? false);
    setShowPassword(false);
    setError(null);
    onClose();
  }, [editUser, onClose]);

  return (
    <Modal isOpen={isOpen} onClose={handleClose} isCentered size="md">
      <ModalOverlay />
      <ModalContent bg="base.800">
        <form onSubmit={handleSubmit}>
          <ModalHeader>{isEdit ? t('auth.userManagement.editUser') : t('auth.userManagement.createUser')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <VStack spacing={4}>
              {!isEdit && (
                <FormControl isRequired>
                  <Grid templateColumns={FORM_GRID_COLUMNS} gap={4} alignItems="start">
                    <GridItem>
                      <FormLabel textAlign="right" mb={0} pt={2}>
                        {t('auth.userManagement.email')}
                      </FormLabel>
                    </GridItem>
                    <GridItem>
                      <Input
                        type="email"
                        value={email}
                        onChange={handleEmailChange}
                        placeholder={t('auth.userManagement.emailPlaceholder')}
                        autoComplete="off"
                      />
                    </GridItem>
                  </Grid>
                </FormControl>
              )}

              <FormControl>
                <Grid templateColumns={FORM_GRID_COLUMNS} gap={4} alignItems="start">
                  <GridItem>
                    <FormLabel textAlign="right" mb={0} pt={2}>
                      {t('auth.userManagement.displayName')}
                    </FormLabel>
                  </GridItem>
                  <GridItem>
                    <Input
                      type="text"
                      value={displayName}
                      onChange={handleDisplayNameChange}
                      placeholder={t('auth.userManagement.displayNamePlaceholder')}
                    />
                  </GridItem>
                </Grid>
              </FormControl>

              <FormControl isInvalid={password.length > 0 && !passwordValidation.isValid} isRequired={!isEdit}>
                <Grid templateColumns={FORM_GRID_COLUMNS} gap={4} alignItems="start">
                  <GridItem>
                    <FormLabel textAlign="right" mb={0} pt={2}>
                      {isEdit ? t('auth.userManagement.newPassword') : t('auth.userManagement.password')}
                    </FormLabel>
                  </GridItem>
                  <GridItem>
                    <InputGroup>
                      <Input
                        type={showPassword ? 'text' : 'password'}
                        value={password}
                        onChange={handlePasswordChange}
                        placeholder={
                          isEdit
                            ? t('auth.userManagement.newPasswordPlaceholder')
                            : t('auth.userManagement.passwordPlaceholder')
                        }
                        autoComplete="new-password"
                        pr="4.5rem"
                      />
                      <InputRightElement w="4.5rem">
                        <Tooltip
                          label={
                            showPassword ? t('auth.userManagement.hidePassword') : t('auth.userManagement.showPassword')
                          }
                        >
                          <IconButton
                            aria-label={
                              showPassword
                                ? t('auth.userManagement.hidePassword')
                                : t('auth.userManagement.showPassword')
                            }
                            icon={showPassword ? <PiEyeSlashBold /> : <PiEyeBold />}
                            variant="ghost"
                            size="sm"
                            onClick={toggleShowPassword}
                            tabIndex={-1}
                          />
                        </Tooltip>
                      </InputRightElement>
                    </InputGroup>
                    {password.length > 0 && !passwordValidation.isValid && (
                      <FormErrorMessage>{passwordValidation.message}</FormErrorMessage>
                    )}
                  </GridItem>
                </Grid>
              </FormControl>

              <Grid templateColumns={FORM_GRID_COLUMNS} gap={4} w="full">
                <GridItem />
                <GridItem>
                  <Button size="sm" variant="ghost" onClick={handleGeneratePassword} leftIcon={<PiLightningFill />}>
                    {t('auth.userManagement.generatePassword')}
                  </Button>
                </GridItem>
              </Grid>

              <FormControl display="flex" alignItems="center">
                <FormLabel mb={0}>{t('auth.userManagement.isAdmin')}</FormLabel>
                <Checkbox isChecked={isAdmin} onChange={handleIsAdminChange} />
              </FormControl>

              {error && (
                <Flex p={3} borderRadius="md" bg="error.500" color="white" fontSize="sm" w="full">
                  <Text fontWeight="semibold">{error}</Text>
                </Flex>
              )}
            </VStack>
          </ModalBody>
          <ModalFooter gap={2}>
            <Button variant="ghost" onClick={handleClose}>
              {t('common.cancel')}
            </Button>
            <Button
              type="submit"
              colorScheme="invokeBlue"
              isLoading={isLoading}
              isDisabled={!isEdit && (!password || !passwordValidation.isValid)}
            >
              {isEdit ? t('common.save') : t('auth.userManagement.createUser')}
            </Button>
          </ModalFooter>
        </form>
      </ModalContent>
    </Modal>
  );
});
UserFormModal.displayName = 'UserFormModal';

// ---------------------------------------------------------------------------
// Delete confirmation modal
// ---------------------------------------------------------------------------

type DeleteUserModalProps = {
  isOpen: boolean;
  onClose: () => void;
  user: UserDTO | null;
};

const DeleteUserModal = memo(({ isOpen, onClose, user }: DeleteUserModalProps) => {
  const { t } = useTranslation();
  const [deleteUser, { isLoading }] = useDeleteUserMutation();
  const [error, setError] = useState<string | null>(null);

  const handleDelete = useCallback(async () => {
    if (!user) {
      return;
    }
    setError(null);
    try {
      await deleteUser(user.user_id).unwrap();
      onClose();
    } catch (err) {
      const detail =
        err && typeof err === 'object' && 'data' in err && typeof (err as { data: unknown }).data === 'object'
          ? ((err as { data: { detail?: string } }).data?.detail ?? t('auth.userManagement.deleteFailed'))
          : t('auth.userManagement.deleteFailed');
      setError(detail);
    }
  }, [user, deleteUser, onClose, t]);

  const handleClose = useCallback(() => {
    setError(null);
    onClose();
  }, [onClose]);

  return (
    <Modal isOpen={isOpen} onClose={handleClose} isCentered size="sm">
      <ModalOverlay />
      <ModalContent bg="base.800">
        <ModalHeader>{t('auth.userManagement.deleteUser')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Text>
            {t('auth.userManagement.deleteConfirm', {
              name: user?.display_name ?? user?.email ?? '',
            })}
          </Text>
          {error && (
            <Flex mt={3} p={3} borderRadius="md" bg="error.500" color="white" fontSize="sm">
              <Text fontWeight="semibold">{error}</Text>
            </Flex>
          )}
        </ModalBody>
        <ModalFooter gap={2}>
          <Button variant="ghost" onClick={handleClose}>
            {t('common.cancel')}
          </Button>
          <Button colorScheme="error" isLoading={isLoading} onClick={handleDelete}>
            {t('common.delete')}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
});
DeleteUserModal.displayName = 'DeleteUserModal';

// ---------------------------------------------------------------------------
// Inline active/inactive toggle
// Wrapping the Switch in a Box lets the Tooltip track mouse-enter/leave
// correctly; without it the tooltip may not dismiss on mouse-out.
// ---------------------------------------------------------------------------

const UserStatusToggle = memo(({ user, isCurrentUser }: { user: UserDTO; isCurrentUser: boolean }) => {
  const { t } = useTranslation();
  const [updateUser, { isLoading }] = useUpdateUserMutation();

  const handleChange = useCallback(
    async (e: ChangeEvent<HTMLInputElement>) => {
      await updateUser({ userId: user.user_id, data: { is_active: e.target.checked } })
        .unwrap()
        .catch(() => null);
    },
    [user.user_id, updateUser]
  );

  const tooltipLabel = isCurrentUser
    ? t('auth.userManagement.cannotDeactivateSelf')
    : user.is_active
      ? t('auth.userManagement.deactivate')
      : t('auth.userManagement.activate');

  return (
    <Tooltip label={tooltipLabel}>
      <Box as="span" display="inline-flex">
        <Switch isChecked={user.is_active} onChange={handleChange} isDisabled={isLoading || isCurrentUser} size="sm" />
      </Box>
    </Tooltip>
  );
});
UserStatusToggle.displayName = 'UserStatusToggle';

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const UserManagement = memo(() => {
  const { t } = useTranslation();
  const currentUser = useAppSelector(selectCurrentUser);
  const navigate = useNavigate();
  const { data: users, isLoading, error } = useListUsersQuery();

  const createModal = useDisclosure();
  const editModal = useDisclosure();
  const deleteModal = useDisclosure();

  const [selectedUser, setSelectedUser] = useState<UserDTO | null>(null);

  const handleBack = useCallback(() => {
    navigate(-1);
  }, [navigate]);

  const handleEdit = useCallback(
    (user: UserDTO) => {
      setSelectedUser(user);
      editModal.onOpen();
    },
    [editModal]
  );

  const handleDelete = useCallback(
    (user: UserDTO) => {
      setSelectedUser(user);
      deleteModal.onOpen();
    },
    [deleteModal]
  );

  const handleEditClose = useCallback(() => {
    editModal.onClose();
    setSelectedUser(null);
  }, [editModal]);

  const handleDeleteClose = useCallback(() => {
    deleteModal.onClose();
    setSelectedUser(null);
  }, [deleteModal]);

  if (isLoading) {
    return (
      <Center py={12}>
        <Spinner size="xl" />
      </Center>
    );
  }

  if (error) {
    return (
      <Center py={12}>
        <Text color="error.400">{t('auth.userManagement.loadFailed')}</Text>
      </Center>
    );
  }

  return (
    <Box p={6}>
      <Flex justify="space-between" align="center" mb={6}>
        <Flex align="center" gap={4}>
          <Button leftIcon={<PiArrowLeftBold />} variant="outline" size="sm" onClick={handleBack}>
            {t('auth.userManagement.back')}
          </Button>
          <Heading size="md">{t('auth.userManagement.title')}</Heading>
        </Flex>
        <Button leftIcon={<PiPlusBold />} colorScheme="invokeBlue" size="sm" onClick={createModal.onOpen}>
          {t('auth.userManagement.createUser')}
        </Button>
      </Flex>

      <Box overflowX="auto">
        <Table variant="simple" size="sm">
          <Thead>
            <Tr>
              <Th>{t('auth.userManagement.email')}</Th>
              <Th>{t('auth.userManagement.displayName')}</Th>
              <Th>{t('auth.userManagement.role')}</Th>
              <Th>{t('auth.userManagement.status')}</Th>
              <Th>{t('auth.userManagement.actions')}</Th>
            </Tr>
          </Thead>
          <Tbody>
            {(users ?? []).map((user) => (
              <UserRow
                key={user.user_id}
                user={user}
                isCurrentUser={user.user_id === currentUser?.user_id}
                onEdit={handleEdit}
                onDelete={handleDelete}
              />
            ))}
          </Tbody>
        </Table>
      </Box>

      {/* Create user modal */}
      <UserFormModal isOpen={createModal.isOpen} onClose={createModal.onClose} />

      {/* Edit user modal */}
      <UserFormModal isOpen={editModal.isOpen} onClose={handleEditClose} editUser={selectedUser} />

      {/* Delete confirmation modal */}
      <DeleteUserModal isOpen={deleteModal.isOpen} onClose={handleDeleteClose} user={selectedUser} />
    </Box>
  );
});
UserManagement.displayName = 'UserManagement';

// ---------------------------------------------------------------------------
// User table row
// ---------------------------------------------------------------------------

type UserRowProps = {
  user: UserDTO;
  isCurrentUser: boolean;
  onEdit: (user: UserDTO) => void;
  onDelete: (user: UserDTO) => void;
};

const UserRow = memo(({ user, isCurrentUser, onEdit, onDelete }: UserRowProps) => {
  const { t } = useTranslation();

  const handleEdit = useCallback(() => {
    onEdit(user);
  }, [user, onEdit]);

  const handleDelete = useCallback(() => {
    onDelete(user);
  }, [user, onDelete]);

  return (
    <Tr>
      <Td>
        <Text fontSize="sm">{user.email}</Text>
        {isCurrentUser && (
          <Badge colorScheme="invokeBlue" size="xs" ml={1}>
            {t('auth.userManagement.you')}
          </Badge>
        )}
      </Td>
      <Td>
        <Text fontSize="sm">{user.display_name ?? '—'}</Text>
      </Td>
      <Td>
        {user.is_admin ? (
          <Badge colorScheme="invokeYellow">{t('auth.admin')}</Badge>
        ) : (
          <Badge colorScheme="base">{t('auth.userManagement.user')}</Badge>
        )}
      </Td>
      <Td>
        <UserStatusToggle user={user} isCurrentUser={isCurrentUser} />
      </Td>
      <Td>
        <Flex gap={1}>
          <Tooltip label={t('auth.userManagement.editUser')}>
            <IconButton
              aria-label={t('auth.userManagement.editUser')}
              icon={<PiPencilBold />}
              variant="ghost"
              size="sm"
              onClick={handleEdit}
            />
          </Tooltip>
          <Tooltip
            label={isCurrentUser ? t('auth.userManagement.cannotDeleteSelf') : t('auth.userManagement.deleteUser')}
          >
            <IconButton
              aria-label={t('auth.userManagement.deleteUser')}
              icon={<PiTrashBold />}
              variant="ghost"
              size="sm"
              colorScheme="error"
              isDisabled={isCurrentUser}
              onClick={handleDelete}
            />
          </Tooltip>
        </Flex>
      </Td>
    </Tr>
  );
});
UserRow.displayName = 'UserRow';
