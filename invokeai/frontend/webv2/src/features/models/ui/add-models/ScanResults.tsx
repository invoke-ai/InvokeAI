/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { FoundModel } from '@features/models/core/types';

import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { InstallSourceButton } from '@features/models/ui/shared/InstallSourceButton';
import { ResultsListHeader } from '@features/models/ui/shared/ResultsListHeader';
import { useInstalledSourceKeys } from '@features/models/ui/shared/useInstalledSources';
import { sourceFileName, sourceLocation, useSourceNameFilter } from '@features/models/ui/shared/useSourceNameFilter';
import { IconButton } from '@platform/ui';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { XIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { InstallOptions } from './InstallOptions';

const pathOf = (result: FoundModel): string => result.path;

export const ScanResults = ({
  fp8Storage,
  inplace,
  onClear,
  onInstall,
  onInstallAll,
  onSetFp8Storage,
  onSetInplace,
  pendingSources,
  scan,
}: {
  fp8Storage: boolean;
  inplace: boolean;
  onClear: () => void;
  onInstall: (path: string) => void;
  /** Bulk path: the parent queues silently and emits one summary toast. */
  onInstallAll: (paths: string[]) => void;
  onSetFp8Storage: (fp8Storage: boolean) => void;
  onSetInplace: (inplace: boolean) => void;
  pendingSources: ReadonlySet<string>;
  scan: { path: string; results: FoundModel[] };
}) => {
  const { t } = useTranslation();
  const { filter, filteredItems: filteredResults, setFilter } = useSourceNameFilter(scan.results, pathOf);
  // Live library state: `is_installed` is a scan-time snapshot, so a model
  // installed from this list would otherwise keep offering Install forever.
  const installedSourceKeys = useInstalledSourceKeys();
  const isRowInstalled = (result: FoundModel) => result.is_installed || installedSourceKeys.has(result.path);

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
          <InstallOptions
            fp8Storage={fp8Storage}
            inplace={inplace}
            onSetFp8Storage={onSetFp8Storage}
            onSetInplace={onSetInplace}
          />
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
      <ListStack dividers label={scan.path}>
        {filteredResults.map((result) => {
          const location = sourceLocation(result.path, scan.path);

          return (
            <ListItem
              key={result.path}
              actions={
                <InstallSourceButton
                  installedModelKey={installedSourceKeys.get(result.path) ?? null}
                  isInstalled={isRowInstalled(result)}
                  isPending={pendingSources.has(result.path)}
                  name={location ?? sourceFileName(result.path)}
                  source={result.path}
                  onInstall={() => onInstall(result.path)}
                />
              }
              description={location ? <MiddleTruncate as="span" text={location} title={result.path} /> : undefined}
              title={sourceFileName(result.path)}
            />
          );
        })}
      </ListStack>
    </Stack>
  );
};
