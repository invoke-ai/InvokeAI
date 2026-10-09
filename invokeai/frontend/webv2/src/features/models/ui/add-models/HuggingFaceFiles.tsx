/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { HFLookupState } from '@features/models/ui/uiStore';

import { Stack } from '@chakra-ui/react';
import {
  InstallSourceContextMenu,
  type InstallSourceStatus,
  useInstallSourceMenu,
} from '@features/models/ui/shared/InstallSourceMenu';
import { InstallPathRow } from '@features/models/ui/shared/InstallSourceRow';
import { ResultsListHeader } from '@features/models/ui/shared/ResultsListHeader';
import { useInstalledSourceKeys } from '@features/models/ui/shared/useInstalledSources';
import { sourceLocation, useSourceNameFilter } from '@features/models/ui/shared/useSourceNameFilter';
import { ListStack } from '@platform/ui/list/ListStack';
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
  const statusOf = (url: string): InstallSourceStatus => ({
    installedModelKey: installedSourceKeys.get(url) ?? null,
    isInstalled: false,
    isPending: pendingSources.has(url),
  });
  const menu = useInstallSourceMenu();
  const menuUrl = menu.shown && filteredUrls.includes(menu.shown.source) ? menu.shown.source : null;

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
        {filteredUrls.map((url) => (
          <InstallPathRow
            key={url}
            {...statusOf(url)}
            isMenuOpen={menuUrl === url}
            location={sourceLocation(url)}
            source={url}
            onInstall={onInstall}
            onOpenMenu={menu.open}
          />
        ))}
      </ListStack>
      <InstallSourceContextMenu
        status={menuUrl === null ? null : statusOf(menuUrl)}
        isOpen={menu.target !== null}
        target={menu.shown}
        onClose={menu.close}
        onExitComplete={menu.release}
        onInstall={() => {
          if (menuUrl !== null) {
            onInstall(menuUrl);
          }
        }}
      />
    </Stack>
  );
};
