import type { WildcardImportEntry, WildcardImportResolution } from '@features/generation/core/wildcardTransfer';
import type { WildcardCatalog } from '@features/generation/ui/useWildcards';
import type { WildcardExportFormat } from '@features/generation/ui/wildcardFiles';
import type { ChangeEvent } from 'react';

import { HStack, Menu, Portal } from '@chakra-ui/react';
import { getWildcardImportActions, planWildcardImport } from '@features/generation/core/wildcardTransfer';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { WildcardImportDialog } from '@features/generation/ui/promptFields/WildcardImportDialog';
import {
  downloadWildcards,
  readWildcardFiles,
  WILDCARD_IMPORT_ACCEPT,
  WildcardFileError,
} from '@features/generation/ui/wildcardFiles';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { DownloadIcon, UploadIcon } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const EXPORT_FORMATS: WildcardExportFormat[] = ['yaml', 'json'];

/**
 * Import and export for the whole catalog.
 *
 * Both directions are pure client work — wildcards are per-user CRUD, so an
 * import is a run of ordinary creates and updates rather than a route of its
 * own. Writes go one at a time and stop at the first failure, so a half-finished
 * import says how far it got instead of reporting success over a silent error.
 */
export const WildcardTransferActions = ({ catalog }: { catalog: WildcardCatalog }) => {
  const { t } = useTranslation();
  const { notifications } = useGenerationUi();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [pendingEntries, setPendingEntries] = useState<WildcardImportEntry[] | null>(null);

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
    async (entries: readonly WildcardImportEntry[], resolutions: Record<string, WildcardImportResolution>) => {
      const actions = getWildcardImportActions(entries, resolutions, new Set(catalog.wildcards.map((w) => w.name)));
      let done = 0;

      setIsBusy(true);

      try {
        for (const action of actions) {
          if (action.id === undefined) {
            await catalog.create({ name: action.name, values: action.values });
          } else {
            await catalog.update(action.id, { name: action.name, values: action.values });
          }
          done++;
        }

        notifications.info(t('widgets.generate.dynamicPrompts.importedCount', { count: done }));
      } catch (caught) {
        // Writes go one at a time, so a failure part-way leaves real wildcards
        // behind. Saying only that the import failed sent people back for a
        // second run that then clashed with everything the first one had made.
        reportError(
          'import-wildcards',
          caught,
          done > 0
            ? t('widgets.generate.dynamicPrompts.couldNotImportAfter', { done, total: actions.length })
            : t('widgets.generate.dynamicPrompts.couldNotImport')
        );
      } finally {
        setIsBusy(false);
        setPendingEntries(null);
      }
    },
    [catalog, notifications, reportError, t]
  );

  const startImport = useCallback(
    async (files: readonly File[]) => {
      setIsBusy(true);

      try {
        const entries = planWildcardImport(await readWildcardFiles(files), catalog.wildcards);

        // Nothing to decide and nothing to explain — the file pick was the
        // confirmation, so a dialog with only an Import button would be a click
        // that tells the user what they already know.
        if (entries.every((entry) => entry.rejection === null && entry.conflictId === null)) {
          await applyImport(entries, {});
          return;
        }

        setPendingEntries(entries);
      } catch (caught) {
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
      try {
        await downloadWildcards(catalog.wildcards, format);
      } catch (caught) {
        reportError('export-wildcards', caught, t('widgets.generate.dynamicPrompts.couldNotExport'));
      }
    },
    [catalog.wildcards, reportError, t]
  );

  const pickFiles = useCallback(() => fileInputRef.current?.click(), []);

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const files = [...(event.currentTarget.files ?? [])];

      if (files.length > 0) {
        void startImport(files);
      }

      event.currentTarget.value = '';
    },
    [startImport]
  );

  const handleExportSelect = useCallback(
    (details: { value: string }) => void runExport(details.value as WildcardExportFormat),
    [runExport]
  );

  const cancelImport = useCallback(() => setPendingEntries(null), []);

  const confirmImport = useCallback(
    (resolutions: Record<string, WildcardImportResolution>) => applyImport(pendingEntries ?? [], resolutions),
    [applyImport, pendingEntries]
  );

  return (
    <HStack gap="0.5">
      <Button disabled={isBusy} size="2xs" variant="ghost" onClick={pickFiles}>
        <UploadIcon />
        {t('widgets.generate.dynamicPrompts.import')}
      </Button>
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
              {EXPORT_FORMATS.map((format) => (
                <Menu.Item key={format} value={format}>
                  <Menu.ItemText fontSize="xs">
                    {t(`widgets.generate.dynamicPrompts.exportAs${format === 'yaml' ? 'Yaml' : 'Json'}`)}
                  </Menu.ItemText>
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
      {pendingEntries ? (
        <WildcardImportDialog entries={pendingEntries} onCancel={cancelImport} onConfirm={confirmImport} />
      ) : null}
    </HStack>
  );
};
