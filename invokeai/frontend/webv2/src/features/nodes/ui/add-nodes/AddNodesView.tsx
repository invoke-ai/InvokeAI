/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Alert, Box, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { validateInstallSource } from '@features/nodes/core/installSource';
import { installCustomNodePack } from '@features/nodes/data/api';
import {
  addCustomNodeInstallLogEntry,
  updateCustomNodeInstallLogEntry,
  useCustomNodeInstallLog,
} from '@features/nodes/data/installLogStore';
import { refreshCustomNodePacks, useCustomNodesSelector } from '@features/nodes/data/nodesStore';
import { updateNodesUi, useNodesUiSelector } from '@features/nodes/ui/nodesUiStore';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, Field, Scrollable } from '@platform/ui';
import { FolderOpenIcon } from 'lucide-react';
import { useMemo, type ChangeEvent, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * One box to add a node pack: a validated Git-URL install plus the manual
 * drop-in path. The typed source lives in the nodes UI store because the
 * detail tabs unmount their content — a tab flip must not lose it — and the
 * busy state is derived from the shared install log so a remount mid-install
 * still shows the install running. Validation mirrors the backend's rules so
 * a doomed source is rejected before the POST.
 */
export const AddNodesView = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const source = useNodesUiSelector((snapshot) => snapshot.installSource);
  const customNodesPath = useCustomNodesSelector((snapshot) => snapshot.customNodesPath);
  const nodePacks = useCustomNodesSelector((snapshot) => snapshot.nodePacks);
  const log = useCustomNodeInstallLog();
  const { isBusy, run } = useScopedAction();

  const trimmedSource = source.trim();
  const installedPackNames = useMemo(() => new Set(nodePacks.map((pack) => pack.name)), [nodePacks]);
  const validation = useMemo(
    () => validateInstallSource(trimmedSource, installedPackNames),
    [installedPackNames, trimmedSource]
  );
  const isInstalling = isBusy || log.some((entry) => entry.status === 'installing');
  const fieldError =
    trimmedSource === '' || validation.issue === null
      ? undefined
      : validation.issue === 'alreadyInstalled'
        ? t('nodes.alreadyInstalledError', { name: validation.packName })
        : t('nodes.invalidSourceName');

  const handleInstall = () => {
    if (validation.issue !== null || isInstalling) {
      return;
    }

    let logEntryId: number | null = null;

    void run(
      async (owner) => {
        // Resolved in place below so the activity badge settles with the install.
        const logEntry = addCustomNodeInstallLogEntry({ name: trimmedSource, status: 'installing' });

        logEntryId = logEntry.id;

        const result = await installCustomNodePack(trimmedSource, owner.signal);

        assertAccountScopeCurrent(owner);

        if (!result.success) {
          updateCustomNodeInstallLogEntry(logEntry.id, { message: result.message, name: result.name, status: 'error' });
          notify.error(t('nodes.installFailedTitle'), result.message);

          return;
        }

        updateCustomNodeInstallLogEntry(logEntry.id, {
          message: result.message,
          name: result.name,
          status: 'completed',
        });
        notify.success(
          t('nodes.installComplete'),
          result.workflows_imported > 0
            ? t('nodes.installCompleteWithWorkflows', { count: result.workflows_imported, name: result.name })
            : result.name
        );
        updateNodesUi({ installSource: '' });
        await refreshCustomNodePacks(owner);
        assertAccountScopeCurrent(owner);

        if (result.requires_dependencies) {
          // Sticky: this demands a manual pip install + restart; a toast that
          // expires quietly buries the instruction.
          notify.warning(
            t('nodes.dependenciesRequired'),
            t('nodes.dependenciesRequiredDescription', {
              dependencyFile: result.dependency_file ?? 'requirements.txt',
              name: result.name,
            }),
            { sticky: true }
          );
        }
      },
      (_message, error) => {
        const message = getApiErrorMessage(error, t('nodes.installFailed'));

        if (logEntryId !== null) {
          updateCustomNodeInstallLogEntry(logEntryId, { message, status: 'error' });
        }

        notify.error(t('nodes.installFailedTitle'), message);
      }
    );
  };

  const handleSourceChange = (event: ChangeEvent<HTMLInputElement>) =>
    updateNodesUi({ installSource: event.currentTarget.value });
  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleInstall();
    }
  };

  return (
    <Scrollable h="full" label={t('nodes.addNodes')} minH="0" p="3">
      <Stack gap="4" maxW="44rem">
        <Alert.Root borderRadius="md" size="sm" status="warning" variant="surface">
          <Alert.Indicator />
          <Alert.Title fontSize="xs">{t('nodes.trustWarning')}</Alert.Title>
        </Alert.Root>
        <Field error={fieldError} helpText={t('nodes.gitUrlHelp')} label={t('nodes.gitUrl')}>
          <HStack align="start" gap="2" w="full">
            <Input
              aria-invalid={fieldError ? true : undefined}
              placeholder="https://github.com/owner/invokeai-node-pack.git"
              size="sm"
              value={source}
              onChange={handleSourceChange}
              onKeyDown={handleKeyDown}
            />
            <Button disabled={validation.issue !== null} loading={isInstalling} size="sm" onClick={handleInstall}>
              {t('nodes.install')}
            </Button>
          </HStack>
        </Field>
        <Stack gap="2">
          <HStack gap="1.5">
            <Icon as={FolderOpenIcon} boxSize="3.5" color="fg.muted" />
            <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
              {t('nodes.installManually')}
            </Text>
          </HStack>
          <Text color="fg.muted" fontSize="xs">
            {t('nodes.scanFolderDescription')}
          </Text>
          {customNodesPath ? (
            <Box bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" p="3" rounded="md">
              <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
                {t('nodes.nodesDirectory')}
              </Text>
              <Text fontFamily="mono" fontSize="xs" mt="1" overflowWrap="anywhere">
                {customNodesPath}
              </Text>
            </Box>
          ) : null}
        </Stack>
      </Stack>
    </Scrollable>
  );
};
