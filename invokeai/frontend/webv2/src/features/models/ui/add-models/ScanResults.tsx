/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { FoundModel } from '@features/models/core/types';

import { Checkbox, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { ResultsListHeader } from '@features/models/ui/shared/ResultsListHeader';
import { InstallSourceButton, SourceListItem } from '@features/models/ui/shared/SourceListItem';
import { useInstalledSources } from '@features/models/ui/shared/useInstalledSources';
import { sourceFileName, useSourceNameFilter } from '@features/models/ui/shared/useSourceNameFilter';
import { IconButton } from '@platform/ui';
import { XIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const pathOf = (result: FoundModel): string => result.path;

export const ScanResults = ({
  inplace,
  onClear,
  onInstall,
  onInstallAll,
  onSetInplace,
  pendingSources,
  scan,
}: {
  inplace: boolean;
  onClear: () => void;
  onInstall: (path: string) => void;
  /** Bulk path: the parent queues silently and emits one summary toast. */
  onInstallAll: (paths: string[]) => void;
  onSetInplace: (inplace: boolean) => void;
  pendingSources: ReadonlySet<string>;
  scan: { path: string; results: FoundModel[] };
}) => {
  const { t } = useTranslation();
  const { filter, filteredItems: filteredResults, setFilter } = useSourceNameFilter(scan.results, pathOf);
  // Live library state: `is_installed` is a scan-time snapshot, so a model
  // installed from this list would otherwise keep offering Install forever.
  const installedSources = useInstalledSources();
  const isRowInstalled = (result: FoundModel) => result.is_installed || installedSources.has(result.path);

  const notInstalledCount = scan.results.filter((result) => !isRowInstalled(result)).length;
  const installable = filteredResults.filter((result) => !isRowInstalled(result));

  const installAll = () => {
    onInstallAll(installable.map((result) => result.path));
  };

  if (scan.results.length === 0) {
    return (
      <HStack justify="space-between">
        <Text color="fg.subtle" fontSize="2xs">
          {t('models.noModelFilesFound', { path: scan.path })}
        </Text>
        <IconButton aria-label={t('models.dismissScanResults')} size="2xs" variant="ghost" onClick={onClear}>
          <Icon as={XIcon} boxSize="3" />
        </IconButton>
      </HStack>
    );
  }

  return (
    <Stack gap="1.5">
      <ResultsListHeader
        extra={
          <Checkbox.Root
            checked={inplace}
            colorPalette="accent"
            size="xs"
            onCheckedChange={(event) => onSetInplace(event.checked === true)}
          >
            <Checkbox.HiddenInput />
            <Checkbox.Control />
            <Checkbox.Label fontSize="2xs">{t('models.installInPlace')}</Checkbox.Label>
          </Checkbox.Root>
        }
        installAllDisabled={installable.length === 0}
        installAllLabel={t('models.installAllCount', { count: installable.length })}
        searchValue={filter}
        summary={t('models.scanSummary', {
          count: scan.results.length,
          notInstalled: notInstalledCount,
          path: scan.path,
        })}
        onClear={onClear}
        onInstallAll={installAll}
        onSearchChange={setFilter}
      />
      {filteredResults.map((result) => (
        <SourceListItem
          key={result.path}
          description={result.path}
          title={sourceFileName(result.path)}
          titleTooltip={result.path}
          trailing={
            <InstallSourceButton
              isInstalled={isRowInstalled(result)}
              isPending={pendingSources.has(result.path)}
              source={result.path}
              onInstall={() => onInstall(result.path)}
            />
          }
        />
      ))}
    </Stack>
  );
};
