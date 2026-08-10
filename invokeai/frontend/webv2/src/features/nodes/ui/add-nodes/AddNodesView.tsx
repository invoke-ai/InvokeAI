import { Alert, Box, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { installCustomNodePack } from '@features/nodes/data/api';
import { addCustomNodeInstallLogEntry, updateCustomNodeInstallLogEntry } from '@features/nodes/data/installLogStore';
import { refreshCustomNodePacks, useCustomNodesSelector } from '@features/nodes/data/nodesStore';
import { updateNodesUi, useNodesUiSelector, type AddNodesTab } from '@features/nodes/ui/nodesUiStore';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, Field, Scrollable, Tabs } from '@platform/ui';
import { FolderOpenIcon, GitBranchIcon } from 'lucide-react';
import { useCallback, useState, type ChangeEvent, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Every way to add custom nodes, one sub-tab per source — Git URL install and
 * the scan-folder workflow. Mirrors the model manager's Add Models view; the
 * active sub-tab lives in the nodes UI store so it survives navigation.
 */
export const AddNodesView = () => {
  const { t } = useTranslation();
  const addTab = useNodesUiSelector((snapshot) => snapshot.addTab);
  const customNodesPath = useCustomNodesSelector((snapshot) => snapshot.customNodesPath);
  const handleValueChange = useCallback(
    (event: { value: string }) => updateNodesUi({ addTab: event.value as AddNodesTab }),
    []
  );

  return (
    <Tabs.Root asChild lazyMount size="sm" unmountOnExit value={addTab} onValueChange={handleValueChange}>
      <Stack gap="3" h="full" minH="0">
        <Tabs.List>
          <Tabs.Trigger value="git">
            <Icon as={GitBranchIcon} boxSize="3" />
            {t('nodes.gitUrl')}
          </Tabs.Trigger>
          <Tabs.Trigger value="scan">
            <Icon as={FolderOpenIcon} boxSize="3" />
            {t('nodes.scanFolder')}
          </Tabs.Trigger>
        </Tabs.List>
        <Box flex="1" minH="0">
          <Tabs.Content h="full" p="0" value="git">
            <Scrollable h="full" label={t('nodes.gitUrl')} minH="0" pr="1">
              <InstallFromGitForm />
            </Scrollable>
          </Tabs.Content>
          <Tabs.Content h="full" p="0" value="scan">
            <Scrollable h="full" label={t('nodes.scanFolder')} minH="0" pr="1">
              <ScanFolderInfo customNodesPath={customNodesPath} />
            </Scrollable>
          </Tabs.Content>
        </Box>
      </Stack>
    </Tabs.Root>
  );
};

const InstallFromGitForm = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const [source, setSource] = useState('');
  const [isInstalling, setIsInstalling] = useState(false);
  const trimmedSource = source.trim();

  const handleInstall = useCallback(async () => {
    if (!trimmedSource) {
      return;
    }

    const owner = captureAccountScope();

    setIsInstalling(true);
    // Resolved in place below so the activity badge settles with the install.
    const logEntry = addCustomNodeInstallLogEntry({ name: trimmedSource, status: 'installing' });

    try {
      const result = await installCustomNodePack(trimmedSource, owner.signal);

      assertAccountScopeCurrent(owner);
      if (result.success) {
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
        setSource('');
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
      } else {
        updateCustomNodeInstallLogEntry(logEntry.id, { message: result.message, name: result.name, status: 'error' });
        notify.error(t('nodes.installFailedTitle'), result.message);
      }
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      const message = getApiErrorMessage(error, t('nodes.installFailed'));

      updateCustomNodeInstallLogEntry(logEntry.id, { message, status: 'error' });
      notify.error(t('nodes.installFailedTitle'), message);
    } finally {
      if (isAccountScopeCurrent(owner)) {
        setIsInstalling(false);
      }
    }
  }, [notify, t, trimmedSource]);
  const handleSourceChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setSource(event.currentTarget.value),
    []
  );
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        void handleInstall();
      }
    },
    [handleInstall]
  );
  const handleInstallClick = useCallback(() => void handleInstall(), [handleInstall]);

  return (
    <Stack gap="4" maxW="44rem">
      <Alert.Root borderRadius="md" size="sm" status="warning" variant="surface">
        <Alert.Indicator />
        <Alert.Title fontSize="xs">{t('nodes.trustWarning')}</Alert.Title>
      </Alert.Root>
      <Field helpText={t('nodes.gitUrlHelp')} label={t('nodes.gitUrl')}>
        <HStack align="start" gap="2">
          <Input
            placeholder="https://github.com/owner/invokeai-node-pack.git"
            size="sm"
            value={source}
            onChange={handleSourceChange}
            onKeyDown={handleKeyDown}
          />
          <Button disabled={!trimmedSource} loading={isInstalling} size="sm" onClick={handleInstallClick}>
            {t('nodes.install')}
          </Button>
        </HStack>
      </Field>
    </Stack>
  );
};

const ScanFolderInfo = ({ customNodesPath }: { customNodesPath: string | null }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="3" maxW="44rem">
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
  );
};
