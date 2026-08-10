/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { HFLookupState } from '@features/models/ui/uiStore';

import { Stack } from '@chakra-ui/react';
import { ResultsListHeader } from '@features/models/ui/shared/ResultsListHeader';
import { InstallSourceButton, SourceListItem } from '@features/models/ui/shared/SourceListItem';
import { useInstalledSources } from '@features/models/ui/shared/useInstalledSources';
import { useDeferredValue, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const fileNameOf = (url: string): string => url.split(/[\\/]/).at(-1) ?? url;

export const HuggingFaceFiles = ({
  lookup,
  onClear,
  onInstall,
  onInstallAll,
  pendingSources,
}: {
  lookup: HFLookupState;
  onClear: () => void;
  onInstall: (url: string) => void;
  /** Bulk path: the parent queues silently and emits one summary toast. */
  onInstallAll: (urls: string[]) => void;
  pendingSources: ReadonlySet<string>;
}) => {
  const { t } = useTranslation();
  const [filter, setFilter] = useState('');
  const deferredFilter = useDeferredValue(filter);
  // A model's recorded install source is the URL it was pulled from, so this
  // marks rows Installed live once the library refresh lands.
  const installedSources = useInstalledSources();

  const filteredUrls = useMemo(() => {
    const term = deferredFilter.trim().toLowerCase();

    if (!term) {
      return lookup.urls;
    }

    return lookup.urls.filter((url) => fileNameOf(url).toLowerCase().includes(term));
  }, [deferredFilter, lookup.urls]);

  const installAll = () => {
    onInstallAll(filteredUrls);
  };

  return (
    <Stack gap="1.5">
      <ResultsListHeader
        installAllDisabled={filteredUrls.length === 0}
        installAllLabel={t('models.installAllCount', { count: filteredUrls.length })}
        searchPlaceholder={t('models.filterFiles')}
        searchValue={filter}
        summary={t('models.filesInRepo', { count: lookup.urls.length, repo: lookup.repo })}
        onClear={onClear}
        onInstallAll={installAll}
        onSearchChange={setFilter}
      />
      {filteredUrls.map((url) => (
        <SourceListItem
          key={url}
          title={fileNameOf(url)}
          titleTooltip={url}
          trailing={
            <InstallSourceButton
              isInstalled={installedSources.has(url)}
              isPending={pendingSources.has(url)}
              source={url}
              onInstall={() => onInstall(url)}
            />
          }
        />
      ))}
    </Stack>
  );
};
