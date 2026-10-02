import type { WildcardImportEntry, WildcardImportResolution } from '@features/generation/core/wildcardTransfer';
import type { WildcardCatalog } from '@features/generation/ui/useWildcards';
import type { WildcardExportFormat, WildcardFileSource } from '@features/generation/ui/wildcardFiles';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ChangeEvent } from 'react';

import { HStack, Menu, Portal } from '@chakra-ui/react';
import { getWildcardImportActions, planWildcardImport } from '@features/generation/core/wildcardTransfer';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { WildcardImportDialog } from '@features/generation/ui/promptFields/WildcardImportDialog';
import { WildcardWriteError } from '@features/generation/ui/useWildcards';
import {
  downloadWildcards,
  WILDCARD_COLLECTION_FORMATS,
  isSupportedWildcardFile,
  readWildcardFiles,
  WILDCARD_IMPORT_ACCEPT,
  WildcardFileError,
} from '@features/generation/ui/wildcardFiles';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { DownloadIcon, UploadIcon } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const IMPORT_SOURCES = ['files', 'folder'] as const;

/** Not in React's DOM typings, and `multiple` comes along with it implicitly. */
const DIRECTORY_INPUT_PROPS = { directory: '', webkitdirectory: '' } as unknown as { webkitdirectory: string };

/** Import CRUD operations sequentially; stop at the first failure and report completed writes. */
export const WildcardTransferActions = ({ catalog }: { catalog: WildcardCatalog }) => {
  const { t } = useTranslation();
  const { notifications } = useGenerationUi();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const directoryInputRef = useRef<HTMLInputElement | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [pendingImport, setPendingImport] = useState<{
    entries: WildcardImportEntry[];
    owner: AccountScope;
  } | null>(null);

  const reportError = useCallback(
    (area: string, caught: unknown, fallback: string) =>
      notifications.reportError({
        area,
        message: getApiErrorMessage(caught, fallback),
        namespace: 'generation',
      }),
    [notifications]
  );

  const applyImport = useCallback(
    async (
      entries: readonly WildcardImportEntry[],
      resolutions: Record<string, WildcardImportResolution>,
      owner: AccountScope
    ) => {
      let actionCount = 0;
      setIsBusy(true);

      try {
        assertAccountScopeCurrent(owner);
        const actions = getWildcardImportActions(entries, resolutions, new Set(catalog.wildcards.map((w) => w.name)));

        actionCount = actions.length;
        notifications.info(
          t('widgets.generate.dynamicPrompts.importedCount', { count: await catalog.applyWrites(actions, owner) })
        );
      } catch (caught) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        // Report partial success so retries do not duplicate completed writes.
        const done = caught instanceof WildcardWriteError ? caught.done : 0;

        reportError(
          'import-wildcards',
          caught,
          done > 0
            ? t('widgets.generate.dynamicPrompts.couldNotImportAfter', { done, total: actionCount })
            : t('widgets.generate.dynamicPrompts.couldNotImport')
        );
      } finally {
        setIsBusy(false);
        setPendingImport(null);
      }
    },
    [catalog, notifications, reportError, t]
  );

  const startImport = useCallback(
    async (files: readonly File[], source: WildcardFileSource) => {
      const owner = captureAccountScope();
      setIsBusy(true);

      try {
        const parsed = await readWildcardFiles(files, source, owner);

        assertAccountScopeCurrent(owner);
        const entries = planWildcardImport(parsed, catalog.wildcards);
        assertAccountScopeCurrent(owner);

        // Import directly when file selection leaves no conflicts or other decisions.
        if (entries.every((entry) => entry.rejection === null && entry.conflictId === null)) {
          await applyImport(entries, {}, owner);
          return;
        }

        setPendingImport({ entries, owner });
      } catch (caught) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        reportError(
          'import-wildcards',
          caught,
          caught instanceof WildcardFileError
            ? t('widgets.generate.dynamicPrompts.couldNotReadFile', { name: caught.fileName })
            : t('widgets.generate.dynamicPrompts.couldNotImport')
        );
      } finally {
        setIsBusy(false);
      }
    },
    [applyImport, catalog.wildcards, reportError, t]
  );

  const runExport = useCallback(
    async (format: WildcardExportFormat) => {
      const owner = captureAccountScope();

      try {
        await downloadWildcards(catalog.wildcards, format, owner);
      } catch (caught) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        reportError('export-wildcards', caught, t('widgets.generate.dynamicPrompts.couldNotExport'));
      }
    },
    [catalog.wildcards, reportError, t]
  );

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const files = [...(event.currentTarget.files ?? [])];

      if (files.length > 0) {
        void startImport(files, 'files');
      }

      event.currentTarget.value = '';
    },
    [startImport]
  );

  // Ignore unrelated directory files, but reject explicitly selected invalid files.
  const handleDirectoryChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const files = [...(event.currentTarget.files ?? [])].filter(isSupportedWildcardFile);

      event.currentTarget.value = '';

      if (files.length === 0) {
        notifications.info(t('widgets.generate.dynamicPrompts.importNoWildcardFiles'));
        return;
      }

      void startImport(files, 'folder');
    },
    [notifications, startImport, t]
  );

  const handleImportSelect = useCallback(
    (details: { value: string }) => (details.value === 'folder' ? directoryInputRef : fileInputRef).current?.click(),
    []
  );

  const handleExportSelect = useCallback(
    (details: { value: string }) => void runExport(details.value as WildcardExportFormat),
    [runExport]
  );

  const cancelImport = useCallback(() => setPendingImport(null), []);

  const confirmImport = useCallback(
    (resolutions: Record<string, WildcardImportResolution>) =>
      pendingImport ? applyImport(pendingImport.entries, resolutions, pendingImport.owner) : Promise.resolve(),
    [applyImport, pendingImport]
  );

  return (
    <HStack gap="0.5">
      <Menu.Root onSelect={handleImportSelect}>
        <Menu.Trigger asChild>
          <Button disabled={isBusy} size="2xs" variant="ghost">
            <UploadIcon />
            {t('widgets.generate.dynamicPrompts.import')}
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="10rem">
              {IMPORT_SOURCES.map((source) => (
                <Menu.Item key={source} value={source}>
                  <Menu.ItemText fontSize="xs">
                    {t(`widgets.generate.dynamicPrompts.import${source === 'folder' ? 'Folder' : 'Files'}`)}
                  </Menu.ItemText>
                </Menu.Item>
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Menu.Root onSelect={handleExportSelect}>
        <Menu.Trigger asChild>
          <Button disabled={isBusy || catalog.wildcards.length === 0} size="2xs" variant="ghost">
            <DownloadIcon />
            {t('widgets.generate.dynamicPrompts.export')}
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="10rem">
              {WILDCARD_COLLECTION_FORMATS.map((format) => (
                <Menu.Item key={format.id} value={format.id}>
                  <Menu.ItemText fontSize="xs">{t(format.labelKey)}</Menu.ItemText>
                </Menu.Item>
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      {/* Multiple, because a wildcard folder is a file per wildcard. */}
      <input
        accept={WILDCARD_IMPORT_ACCEPT}
        hidden
        multiple
        ref={fileInputRef}
        type="file"
        onChange={handleFileChange}
      />
      {/* Use a separate directory input to preserve relative wildcard paths. */}
      <input {...DIRECTORY_INPUT_PROPS} hidden ref={directoryInputRef} type="file" onChange={handleDirectoryChange} />
      {pendingImport ? (
        <WildcardImportDialog entries={pendingImport.entries} onCancel={cancelImport} onConfirm={confirmImport} />
      ) : null}
    </HStack>
  );
};
