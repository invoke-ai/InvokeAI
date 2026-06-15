import { Avatar, Badge, Box, Center, Flex, HStack, Spinner, Stack, Switch, Table, Text } from '@chakra-ui/react';
import { useCallback, useEffect, useState } from 'react';
import { PencilIcon, Trash2Icon, UserPlusIcon } from 'lucide-react';

import { deleteUser, listUsers, updateUser, type UserDTO } from '@workbench/auth/api';
import { useAuthSession } from '@workbench/auth/session';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button, IconButton } from '@workbench/components/ui/Button';
import { ConfirmDialog } from '@workbench/components/ui/ConfirmDialog';
import { Scrollable } from '@workbench/components/ui/Scrollable';
import { Tooltip } from '@workbench/components/ui/Tooltip';
import { useNotify } from '@workbench/useNotify';
import { UserFormDialog, type UserFormTarget } from './UserFormDialog';

/**
 * Admin-only center view: the workspace user directory with create, edit,
 * activate/deactivate, and delete. The widget is already hidden for
 * non-admins via `requiresAdmin`; the inline guard covers stale persisted
 * layouts that still point at it.
 */
export const UserManagement = () => {
  return (
    <Scrollable flex="1" h="full" label="User management">
      <UsersManagementPanel />
    </Scrollable>
  );
};

export const UsersManagementPanel = () => {
  const session = useAuthSession();

  if (!session.multiuserEnabled || session.user?.is_admin !== true) {
    return (
      <Center h="full" w="full">
        <Text color="fg.subtle" fontSize="sm">
          User management is available to administrators only.
        </Text>
      </Center>
    );
  }

  return <UsersDirectory currentUserId={session.user.user_id} />;
};

const UsersDirectory = ({ currentUserId }: { currentUserId: string }) => {
  const notify = useNotify();
  const [users, setUsers] = useState<UserDTO[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [formTarget, setFormTarget] = useState<UserFormTarget | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<UserDTO | null>(null);

  const refresh = useCallback(async () => {
    try {
      setUsers(await listUsers());
      setLoadError(null);
    } catch (error) {
      setLoadError(getApiErrorMessage(error, 'Could not load users.'));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const setUserActive = async (user: UserDTO, isActive: boolean) => {
    try {
      await updateUser(user.user_id, { is_active: isActive });
      await refresh();
    } catch (error) {
      notify.error(
        isActive ? 'Could not activate the user' : 'Could not deactivate the user',
        getApiErrorMessage(error, 'The backend rejected the change.')
      );
    }
  };

  const confirmDelete = async () => {
    if (!deleteTarget) {
      return;
    }

    try {
      await deleteUser(deleteTarget.user_id);
      notify.success('User deleted');
      await refresh();
    } catch (error) {
      notify.error('Could not delete the user', getApiErrorMessage(error, 'The backend rejected the request.'));
    }
  };

  return (
    <>
      <Flex justify="center" px="4" py="6" w="full">
        <Stack gap="4" w="full">
          <HStack align="flex-start" justify="space-between">
            <Stack gap="0.5">
              <Text fontSize="md" fontWeight="700">
                Users
              </Text>
              <Text color="fg.subtle" fontSize="xs">
                {users
                  ? `${users.length} ${users.length === 1 ? 'account' : 'accounts'} in this workspace.`
                  : 'Manage workspace accounts and roles.'}
              </Text>
            </Stack>
            <Button size="xs" variant="solid" onClick={() => setFormTarget({ mode: 'create' })}>
              <UserPlusIcon />
              Add user
            </Button>
          </HStack>
          <Box bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" overflowX="auto" rounded="lg">
            {users === null ? (
              <Center minH="40">
                {loadError ? (
                  <Stack align="center" gap="2" p="4">
                    <Text color="fg.error" fontSize="xs" textAlign="center">
                      {loadError}
                    </Text>
                    <Button size="xs" variant="outline" onClick={() => void refresh()}>
                      Retry
                    </Button>
                  </Stack>
                ) : (
                  <Spinner color="fg.muted" size="sm" />
                )}
              </Center>
            ) : (
              <Table.Root minW="42rem" size="sm">
                <Table.Header>
                  <Table.Row bg="bg.muted">
                    <Table.ColumnHeader borderColor="border.subtle" color="fg.muted" ps="4">
                      User
                    </Table.ColumnHeader>
                    <Table.ColumnHeader borderColor="border.subtle" color="fg.muted">
                      Role
                    </Table.ColumnHeader>
                    <Table.ColumnHeader borderColor="border.subtle" color="fg.muted">
                      Last sign-in
                    </Table.ColumnHeader>
                    <Table.ColumnHeader borderColor="border.subtle" color="fg.muted">
                      Active
                    </Table.ColumnHeader>
                    <Table.ColumnHeader borderColor="border.subtle" pe="4" />
                  </Table.Row>
                </Table.Header>
                <Table.Body>
                  {users.map((user) => (
                    <UserRow
                      key={user.user_id}
                      isSelf={user.user_id === currentUserId}
                      user={user}
                      onDelete={() => setDeleteTarget(user)}
                      onEdit={() => setFormTarget({ mode: 'edit', user })}
                      onSetActive={(isActive) => void setUserActive(user, isActive)}
                    />
                  ))}
                </Table.Body>
              </Table.Root>
            )}
          </Box>
        </Stack>
      </Flex>
      <UserFormDialog target={formTarget} onClose={() => setFormTarget(null)} onSaved={() => void refresh()} />
      <ConfirmDialog
        body={`Delete ${deleteTarget ? getUserLabel(deleteTarget) : 'this user'}? Their account is removed permanently.`}
        confirmLabel="Delete user"
        isOpen={deleteTarget !== null}
        title="Delete user?"
        onClose={() => setDeleteTarget(null)}
        onConfirm={confirmDelete}
      />
    </>
  );
};

const getUserLabel = (user: UserDTO): string => user.display_name?.trim() || user.email;

const formatLastSignIn = (lastLoginAt: string | null): string => {
  if (!lastLoginAt) {
    return 'Never';
  }

  const date = new Date(lastLoginAt);

  return Number.isNaN(date.getTime()) ? 'Never' : date.toLocaleString();
};

const UserRow = ({
  isSelf,
  onDelete,
  onEdit,
  onSetActive,
  user,
}: {
  isSelf: boolean;
  onDelete: () => void;
  onEdit: () => void;
  onSetActive: (isActive: boolean) => void;
  user: UserDTO;
}) => (
  <Table.Row bg="transparent" _hover={{ bg: 'bg.muted' }}>
    <Table.Cell borderColor="border.subtle" ps="4">
      <HStack gap="2.5">
        <Avatar.Root bg="accent.subtle" color="fg" size="2xs">
          <Avatar.Fallback fontSize="2xs" name={getUserLabel(user)} />
        </Avatar.Root>
        <Stack gap="0">
          <HStack gap="1.5">
            <Text fontSize="xs" fontWeight="600">
              {getUserLabel(user)}
            </Text>
            {isSelf ? (
              <Badge fontSize="2xs" variant="surface">
                You
              </Badge>
            ) : null}
          </HStack>
          <Text color="fg.muted" fontSize="2xs">
            {user.email}
          </Text>
        </Stack>
      </HStack>
    </Table.Cell>
    <Table.Cell borderColor="border.subtle">
      <Badge colorPalette={user.is_admin ? 'purple' : 'gray'} fontSize="2xs" variant="surface">
        {user.is_admin ? 'Admin' : 'User'}
      </Badge>
    </Table.Cell>
    <Table.Cell borderColor="border.subtle">
      <Text color="fg.muted" fontSize="2xs">
        {formatLastSignIn(user.last_login_at)}
      </Text>
    </Table.Cell>
    <Table.Cell borderColor="border.subtle">
      <Tooltip content="You cannot deactivate your own account." disabled={!isSelf} showArrow>
        <Switch.Root
          aria-label={`${getUserLabel(user)} active`}
          checked={user.is_active}
          disabled={isSelf}
          size="sm"
          onCheckedChange={(event) => onSetActive(event.checked)}
        >
          <Switch.HiddenInput />
          <Switch.Control _checked={{ bg: 'accent.solid' }}>
            <Switch.Thumb />
          </Switch.Control>
        </Switch.Root>
      </Tooltip>
    </Table.Cell>
    <Table.Cell borderColor="border.subtle" pe="4" textAlign="end">
      <HStack gap="0.5" justify="flex-end">
        <IconButton
          aria-label={`Edit ${getUserLabel(user)}`}
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={onEdit}
        >
          <PencilIcon />
        </IconButton>
        <Tooltip content="You cannot delete your own account." disabled={!isSelf} showArrow>
          <IconButton
            aria-label={`Delete ${getUserLabel(user)}`}
            color="fg.muted"
            disabled={isSelf}
            size="2xs"
            variant="ghost"
            _hover={{ color: 'fg.error' }}
            onClick={onDelete}
          >
            <Trash2Icon />
          </IconButton>
        </Tooltip>
      </HStack>
    </Table.Cell>
  </Table.Row>
);
