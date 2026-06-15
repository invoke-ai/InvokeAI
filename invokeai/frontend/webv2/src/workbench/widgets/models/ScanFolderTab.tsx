import { Checkbox, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui/Button';
import { Field } from '@workbench/components/ui/Field';
import { Scrollable } from '@workbench/components/ui/Scrollable';
import { scanFolderForModels } from '@workbench/models/api';
import { scanFolderSchema } from '@workbench/models/schemas';
import { updateModelsUi, useModelsUi } from '@workbench/models/uiStore';
import { useZodForm } from '@workbench/models/useZodForm';
import { useNotify } from '@workbench/useNotify';
import { DownloadIcon, FolderSearchIcon } from 'lucide-react';
import { useState } from 'react';

import { InstallSourceButton, SourceListItem } from './SourceListItem';
import { useInstallActions } from './useInstallActions';

/**
 * Recursively scan a server folder for model files and install them
 * individually or all at once, optionally in place. The query and results
 * live in the models UI store, so leaving and revisiting the tab keeps them.
 */
export const ScanFolderTab = () => {
  const notify = useNotify();
  const { install, pendingSources } = useInstallActions();
  const { scan } = useModelsUi();
  const [inplace, setInplace] = useState(true);
  const [isInstallingAll, setIsInstallingAll] = useState(false);
  const form = useZodForm(scanFolderSchema, { path: scan?.path ?? '' });

  const handleScan = () =>
    form.handleSubmit(async ({ path }) => {
      updateModelsUi({ scan: { path, results: await scanFolderForModels(path) } });
    });

  const notInstalled = scan?.results.filter((result) => !result.is_installed) ?? [];

  const handleInstallAll = async () => {
    setIsInstallingAll(true);

    try {
      let queued = 0;

      for (const result of notInstalled) {
        // Sequential on purpose: each call returns fast (install is a
        // background job) and ordering keeps the queue readable.
        if (await install({ inplace, source: result.path }, { silent: true })) {
          queued += 1;
        }
      }

      if (queued > 0) {
        notify.success('Installs queued', `${queued} model${queued === 1 ? '' : 's'} queued from the scanned folder.`);
      }
    } finally {
      setIsInstallingAll(false);
    }
  };

  return (
    <Stack gap="4" maxW="40rem">
      <Field
        error={form.errors.path ?? form.formError}
        helpText="An absolute folder path on the machine running InvokeAI. Subfolders are scanned too."
        label="Folder to Scan"
      >
        <HStack gap="1.5">
          <Input
            aria-invalid={form.errors.path ? true : undefined}
            placeholder="/path/to/models"
            size="sm"
            value={form.values.path}
            onChange={(event) => form.setValue('path', event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                void handleScan();
              }
            }}
          />
          <Button
            disabled={form.values.path.trim().length === 0}
            loading={form.isSubmitting}
            size="sm"
            variant="solid"
            onClick={() => void handleScan()}
          >
            <Icon as={FolderSearchIcon} boxSize="3.5" />
            Scan
          </Button>
        </HStack>
      </Field>

      {scan ? (
        <Stack gap="2">
          <HStack justify="space-between" wrap="wrap">
            <Text color="fg.muted" fontSize="2xs">
              {scan.results.length} model file{scan.results.length === 1 ? '' : 's'} found · {notInstalled.length} not
              installed
            </Text>
            <HStack gap="3">
              <Checkbox.Root
                checked={inplace}
                colorPalette="accent"
                size="sm"
                onCheckedChange={(event) => setInplace(event.checked === true)}
              >
                <Checkbox.HiddenInput />
                <Checkbox.Control />
                <Checkbox.Label fontSize="2xs">Install in place</Checkbox.Label>
              </Checkbox.Root>
              <Button
                disabled={notInstalled.length === 0}
                loading={isInstallingAll}
                size="2xs"
                variant="solid"
                onClick={() => void handleInstallAll()}
              >
                <Icon as={DownloadIcon} boxSize="3" />
                Install All ({notInstalled.length})
              </Button>
            </HStack>
          </HStack>
          {scan.results.length === 0 ? (
            <Text color="fg.subtle" fontSize="2xs">
              No model files found in {scan.path}.
            </Text>
          ) : (
            <Scrollable label="Scan results" maxH="50vh">
              <Stack gap="1.5">
                {scan.results.map((result) => (
                  <SourceListItem
                    key={result.path}
                    description={result.path}
                    title={result.path.split('/').at(-1) ?? result.path}
                    titleTooltip={result.path}
                    trailing={
                      <InstallSourceButton
                        isInstalled={result.is_installed}
                        isPending={pendingSources.has(result.path)}
                        source={result.path}
                        onInstall={() => {
                          void install({ inplace, source: result.path });
                        }}
                      />
                    }
                  />
                ))}
              </Stack>
            </Scrollable>
          )}
        </Stack>
      ) : null}
    </Stack>
  );
};
