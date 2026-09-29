/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { HFLookupState } from '@features/models/ui/uiStore';

import { Stack } from '@chakra-ui/react';
import { InstallSourceButton } from '@features/models/ui/shared/InstallSourceButton';
import { ResultsListHeader } from '@features/models/ui/shared/ResultsListHeader';
import { useInstalledSourceKeys } from '@features/models/ui/shared/useInstalledSources';
import { sourceFileName, sourceLocation, useSourceNameFilter } from '@features/models/ui/shared/useSourceNameFilter';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useTranslation } from 'react-i18next';

import { InstallOptions } from './InstallOptions';

const urlOf = (url: string): string => url;

export const HuggingFaceFiles = ({
  fp8Storage,
  lookup,
  onClear,
  onInstall,
  onInstallAll,
  onSetFp8Storage,
  pendingSources,
}: {
  fp8Storage: boolean;
  lookup: HFLookupState;
  onClear: () => void;
  onInstall: (url: string) => void;
  /** Bulk path: the parent queues silently and emits one summary toast. */
  onInstallAll: (urls: string[]) => void;
  onSetFp8Storage: (fp8Storage: boolean) => void;
  pendingSources: ReadonlySet<string>;
}) => {
  const { t } = useTranslation();
  const { filter, filteredItems: filteredUrls, setFilter } = useSourceNameFilter(lookup.urls, urlOf);
  const installedSourceKeys = useInstalledSourceKeys();

  const installAll = () => {
    onInstallAll([...filteredUrls]);
  };

  return (
    <Stack gap="1.5">
      <ResultsListHeader
        extra={<InstallOptions fp8Storage={fp8Storage} onSetFp8Storage={onSetFp8Storage} />}
        installAllDisabled={filteredUrls.length === 0}
        installAllLabel={t('models.installAllCount', { count: filteredUrls.length })}
        searchPlaceholder={t('models.filterFiles')}
        searchValue={filter}
        summary={t('models.filesInRepo', { count: lookup.urls.length, repo: lookup.repo })}
        onClear={onClear}
        onInstallAll={installAll}
        onSearchChange={setFilter}
      />
      <ListStack dividers label={lookup.repo}>
        {filteredUrls.map((url) => {
          const location = sourceLocation(url);

          return (
            <ListItem
              key={url}
              actions={
                <InstallSourceButton
                  installedModelKey={installedSourceKeys.get(url) ?? null}
                  isPending={pendingSources.has(url)}
                  name={location ?? sourceFileName(url)}
                  source={url}
                  onInstall={() => onInstall(url)}
                />
              }
              description={location ? <MiddleTruncate as="span" text={location} title={url} /> : undefined}
              title={sourceFileName(url)}
            />
          );
        })}
      </ListStack>
    </Stack>
  );
};
