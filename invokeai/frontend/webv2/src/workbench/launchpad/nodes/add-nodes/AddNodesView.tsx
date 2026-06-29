import { Alert, Box, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { getApiErrorMessage } from '@workbench/backend/http';
import { Button, Field, Scrollable, Tabs, toaster } from '@workbench/components/ui';
import { installCustomNodePack } from '@workbench/customNodes/api';
import { addCustomNodeInstallLogEntry } from '@workbench/customNodes/installLogStore';
import { refreshCustomNodePacks, useCustomNodesSelector } from '@workbench/customNodes/nodesStore';
import { updateNodesUi, useNodesUiSelector, type AddNodesTab } from '@workbench/customNodes/nodesUiStore';
import { FolderOpenIcon, GitBranchIcon } from 'lucide-react';
import { useCallback, useState, type ChangeEvent, type KeyboardEvent } from 'react';

/**
 * Every way to add custom nodes, one sub-tab per source — Git URL install and
 * the scan-folder workflow. Mirrors the model manager's Add Models view; the
 * active sub-tab lives in the nodes UI store so it survives navigation.
 */
export const AddNodesView = () => {
  const addTab = useNodesUiSelector((snapshot) => snapshot.addTab);
  const customNodesPath = useCustomNodesSelector((snapshot) => snapshot.customNodesPath);
  const handleValueChange = useCallback(
    (event: { value: string }) => updateNodesUi({ addTab: event.value as AddNodesTab }),
    []
  );

  return (
    <Stack gap="3" h="full" minH="0">
      <Tabs.Root size="sm" value={addTab} onValueChange={handleValueChange}>
        <Tabs.List>
          <Tabs.Trigger fontSize="xs" value="git">
            <Icon as={GitBranchIcon} boxSize="3" />
            Git URL
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="scan">
            <Icon as={FolderOpenIcon} boxSize="3" />
            Scan Folder
          </Tabs.Trigger>
        </Tabs.List>
      </Tabs.Root>
      <Scrollable flex="1" label="Add custom nodes" minH="0" pr="1">
        {addTab === 'git' ? <InstallFromGitForm /> : <ScanFolderInfo customNodesPath={customNodesPath} />}
      </Scrollable>
    </Stack>
  );
};

const InstallFromGitForm = () => {
  const [source, setSource] = useState('');
  const [isInstalling, setIsInstalling] = useState(false);
  const trimmedSource = source.trim();

  const handleInstall = useCallback(async () => {
    if (!trimmedSource) {
      return;
    }

    setIsInstalling(true);
    addCustomNodeInstallLogEntry({ name: trimmedSource, status: 'installing' });

    try {
      const result = await installCustomNodePack(trimmedSource);

      if (result.success) {
        addCustomNodeInstallLogEntry({ message: result.message, name: result.name, status: 'completed' });
        setSource('');
        await refreshCustomNodePacks();

        if (result.requires_dependencies) {
          toaster.create({
            description: `${result.name} requires dependencies from ${result.dependency_file ?? 'requirements.txt'}. Install them, then restart InvokeAI.`,
            title: 'Dependencies required',
            type: 'warning',
          });
        }
      } else {
        addCustomNodeInstallLogEntry({ message: result.message, name: result.name, status: 'error' });
      }
    } catch (error) {
      addCustomNodeInstallLogEntry({
        message: getApiErrorMessage(error, 'Install failed.'),
        name: trimmedSource,
        status: 'error',
      });
    } finally {
      setIsInstalling(false);
    }
  }, [trimmedSource]);
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
        <Alert.Title fontSize="xs">
          Custom nodes execute Python code. Only install packs from authors you trust.
        </Alert.Title>
      </Alert.Root>
      <Field
        helpText="Paste a public Git repository URL. InvokeAI clones it into your custom nodes directory."
        label="Git URL"
      >
        <HStack align="start" gap="2">
          <Input
            placeholder="https://github.com/owner/invokeai-node-pack.git"
            size="sm"
            value={source}
            onChange={handleSourceChange}
            onKeyDown={handleKeyDown}
          />
          <Button disabled={!trimmedSource} loading={isInstalling} size="sm" onClick={handleInstallClick}>
            Install
          </Button>
        </HStack>
      </Field>
    </Stack>
  );
};

const ScanFolderInfo = ({ customNodesPath }: { customNodesPath: string | null }) => (
  <Stack gap="3" maxW="44rem">
    <Text color="fg.muted" fontSize="xs">
      Add node packs directly to the custom nodes directory, then reload nodes. InvokeAI scans this folder on startup
      and reload.
    </Text>
    {customNodesPath ? (
      <Box bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" p="3" rounded="md">
        <Text color="fg.subtle" fontSize="2xs" fontWeight="600" textTransform="uppercase">
          Nodes Directory
        </Text>
        <Text fontFamily="mono" fontSize="xs" mt="1" overflowWrap="anywhere">
          {customNodesPath}
        </Text>
      </Box>
    ) : null}
  </Stack>
);
