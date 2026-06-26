/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { HFLookupState } from '@workbench/models/uiStore';

import { Stack } from '@chakra-ui/react';
import { ResultsListHeader } from '@workbench/launchpad/models/shared/ResultsListHeader';
import { InstallSourceButton, SourceListItem } from '@workbench/launchpad/models/shared/SourceListItem';
import { useDeferredValue, useMemo, useState } from 'react';

const fileNameOf = (url: string): string => url.split(/[\\/]/).at(-1) ?? url;

export const HuggingFaceFiles = ({
  lookup,
  onClear,
  onInstall,
  pendingSources,
}: {
  lookup: HFLookupState;
  onClear: () => void;
  onInstall: (url: string) => void;
  pendingSources: ReadonlySet<string>;
}) => {
  const [filter, setFilter] = useState('');
  const deferredFilter = useDeferredValue(filter);

  const filteredUrls = useMemo(() => {
    const term = deferredFilter.trim().toLowerCase();

    if (!term) {
      return lookup.urls;
    }

    return lookup.urls.filter((url) => fileNameOf(url).toLowerCase().includes(term));
  }, [deferredFilter, lookup.urls]);

  const installAll = () => {
    for (const url of filteredUrls) {
      onInstall(url);
    }
  };

  return (
    <Stack gap="1.5">
      <ResultsListHeader
        installAllDisabled={filteredUrls.length === 0}
        installAllLabel={`Install all (${filteredUrls.length})`}
        searchPlaceholder="Filter files"
        searchValue={filter}
        summary={`${lookup.urls.length} file${lookup.urls.length === 1 ? '' : 's'} in ${lookup.repo}`}
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
            <InstallSourceButton isPending={pendingSources.has(url)} source={url} onInstall={() => onInstall(url)} />
          }
        />
      ))}
    </Stack>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
