/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ListRowProps } from '@platform/ui/list/List';

import { Box, Flex, HStack, Icon, Text } from '@chakra-ui/react';
import { List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { listRowsFromSections, type ListRow } from '@platform/ui/list/listRows';
import { ManagerColumn, ManagerDetailHeader } from '@platform/ui/ManagerLayout';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { Navigate, useNavigate, useParams, useSearch } from '@tanstack/react-router';
import { SearchIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { SettingsSection } from './catalog';

import { browseSettings } from './catalog';
import { SettingsBrowseBody, SettingsSearchField } from './SettingsBrowseBody';
import { focusSettingsSearchOnSlash } from './settingsSearchShortcut';
import { useAvailableSettings } from './useAvailableSettings';

const GROUPS = ['application', 'project', 'widgets', 'system'] as const;
const ALL_RESULTS_KEY = 'all-results';

type ListEntry = { kind: 'all' } | { kind: 'section'; section: SettingsSection };

/**
 * Launchpad preferences: the settings that need no open project, in the managers' list/detail layout. The
 * section lives in the route; the query stays in the page, which the Launchpad keeps mounted between visits.
 */
export const PreferencesPage = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  // Undefined on bare `/preferences` and on the other Launchpad pages this stays mounted behind.
  const requestedId = useParams({ select: (params: { section?: string }) => params.section, strict: false }) ?? null;
  const revealEntryId = useSearch({ select: (search: { setting?: string }) => search.setting, strict: false });
  const [query, setQuery] = useState('');
  const [searchSection, setSearchSection] = useState<string | null>(null);
  const [lastSectionId, setLastSectionId] = useState<string | null>(null);
  const [lastRequest, setLastRequest] = useState({ requestedId, revealEntryId });
  const sections = useAvailableSettings();
  const isKnownRequest = requestedId !== null && sections.some((section) => section.id === requestedId);

  // Bare `/preferences` (the rail link) returns to the section last shown.
  if (isKnownRequest && requestedId !== lastSectionId) {
    setLastSectionId(requestedId);
  }
  // A section or setting requested from elsewhere (an entry point, the palette) replaces a search left behind.
  if (lastRequest.requestedId !== requestedId || lastRequest.revealEntryId !== revealEntryId) {
    setLastRequest({ requestedId, revealEntryId });
    if (
      (requestedId !== null && requestedId !== lastRequest.requestedId) ||
      (revealEntryId !== undefined && revealEntryId !== lastRequest.revealEntryId)
    ) {
      setQuery('');
      setSearchSection(null);
    }
  }

  const { active, count, displayed, matches, searching } = browseSettings(
    sections,
    { activeId: (isKnownRequest ? requestedId : lastSectionId) ?? '', query, searchSection },
    t
  );

  const changeQuery = (next: string) => {
    setQuery(next);
    setSearchSection(null);
  };
  const clearSearch = () => changeQuery('');
  const openSection = (sectionId: string, entryId?: string) => {
    void navigate({
      params: { section: sectionId },
      search: entryId ? { setting: entryId } : {},
      to: '/preferences/$section',
    });
  };
  const reveal = (sectionId: string, entryId?: string) => {
    changeQuery('');
    openSection(sectionId, entryId);
  };
  const clearRevealedEntry = () => {
    void navigate({ params: { section: active.id }, replace: true, search: {}, to: '/preferences/$section' });
  };

  const sectionRows = listRowsFromSections<ListEntry>(
    GROUPS.map((group) => ({
      items: matches
        .filter((section) => section.group === group)
        .map((section) => ({ kind: 'section', section }) as const),
      key: group,
      label: t(`settingsDialog.groups.${group}`),
    })),
    (entry) => (entry.kind === 'all' ? ALL_RESULTS_KEY : entry.section.id)
  ).map((row) =>
    // While searching, group headers count matching settings like the rows do, not sections.
    searching && row.kind === 'header'
      ? {
          ...row,
          count: matches
            .filter((section) => `header:${section.group}` === row.key)
            .reduce((total, section) => total + section.entries.length, 0),
        }
      : row
  );
  // While searching, "All results" leads the list on its own, above the grouped sections.
  const rows: ListRow<ListEntry>[] = searching
    ? [{ item: { kind: 'all' }, key: ALL_RESULTS_KEY, kind: 'item' }, ...sectionRows]
    : sectionRows;

  const renderItem = (entry: ListEntry, rowProps: ListRowProps) =>
    entry.kind === 'all' ? (
      <ListItem
        {...rowProps}
        leading={<Icon as={SearchIcon} boxSize="3.5" />}
        title={t('settingsDialog.allResults')}
        titleTruncate="end"
        trailing={String(count)}
        onPress={() => setSearchSection(null)}
      />
    ) : (
      <ListItem
        {...rowProps}
        leading={<Icon as={entry.section.icon} boxSize="3.5" />}
        title={resolveSettingsText(entry.section.label, t)}
        titleTruncate="end"
        trailing={searching ? String(entry.section.entries.length) : undefined}
        onPress={() => (searching ? setSearchSection(entry.section.id) : openSection(entry.section.id))}
      />
    );

  if (requestedId !== null && !isKnownRequest) {
    return <Navigate params={{ section: active.id }} replace to="/preferences/$section" />;
  }

  return (
    <Flex
      aria-label={t('launchpad.sections.preferences')}
      role="region"
      h="full"
      minH="0"
      w="full"
      onKeyDown={focusSettingsSearchOnSlash}
    >
      <ManagerColumn title={t('launchpad.sections.preferences')}>
        <Box p="3">
          <SettingsSearchField size="xs" value={query} onChange={changeQuery} />
        </Box>
        <List
          activeKey={searching ? (searchSection ?? ALL_RESULTS_KEY) : active.id}
          density="compact"
          label={t('settingsDialog.section')}
          renderItem={renderItem}
          rows={rows}
          status="ready"
        />
      </ManagerColumn>
      <Flex direction="column" flex="1" minH="0" minW="0">
        <ManagerDetailHeader>
          <HStack alignSelf="stretch" gap="2" px="1">
            <Icon as={searching ? SearchIcon : active.icon} boxSize="4" />
            <Text as="h2" fontSize="sm" fontWeight="700">
              {searching ? t('settingsDialog.results') : resolveSettingsText(active.label, t)}
            </Text>
          </HStack>
        </ManagerDetailHeader>
        <SettingsBrowseBody
          count={count}
          displayed={displayed}
          revealEntryId={revealEntryId}
          searching={searching}
          viewKey={searching ? `search:${searchSection ?? ''}` : active.id}
          onClearSearch={clearSearch}
          onReveal={reveal}
          onRevealed={clearRevealedEntry}
        />
      </Flex>
    </Flex>
  );
};
