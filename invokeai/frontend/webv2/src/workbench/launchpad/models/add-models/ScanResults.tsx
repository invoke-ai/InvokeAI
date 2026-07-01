/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { FoundModel } from '@workbench/models/types';

import { Checkbox, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { IconButton } from '@workbench/components/ui';
import { ResultsListHeader } from '@workbench/launchpad/models/shared/ResultsListHeader';
import { InstallSourceButton, SourceListItem } from '@workbench/launchpad/models/shared/SourceListItem';
import { XIcon } from 'lucide-react';
import { useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const fileNameOf = (path: string): string => path.split(/[\\/]/).at(-1) ?? path;

export const ScanResults = ({
  inplace,
  onClear,
  onInstall,
  onSetInplace,
  pendingSources,
  scan,
}: {
  inplace: boolean;
  onClear: () => void;
  onInstall: (path: string) => void;
  onSetInplace: (inplace: boolean) => void;
  pendingSources: ReadonlySet<string>;
  scan: { path: string; results: FoundModel[] };
}) => {
  const { t } = useTranslation();
  const [filter, setFilter] = useState('');
  const deferredFilter = useDeferredValue(filter);

  const filteredResults = useMemo(() => {
    const term = deferredFilter.trim().toLowerCase();

    if (!term) {
      return scan.results;
    }

    return scan.results.filter((result) => fileNameOf(result.path).toLowerCase().includes(term));
  }, [deferredFilter, scan.results]);

  const notInstalledCount = scan.results.filter((result) => !result.is_installed).length;
  const installable = filteredResults.filter((result) => !result.is_installed);

  const installAll = () => {
    for (const result of installable) {
      onInstall(result.path);
    }
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
          title={fileNameOf(result.path)}
          titleTooltip={result.path}
          trailing={
            <InstallSourceButton
              isInstalled={result.is_installed}
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
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
