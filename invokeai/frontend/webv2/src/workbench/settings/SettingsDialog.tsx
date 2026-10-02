import type { ChangeEvent, UIEvent } from 'react';

import { Box, Dialog, Flex, HStack, Icon, NativeSelect, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { PanelHeader } from '@platform/ui/PanelHeader';
import { Scrollable } from '@platform/ui/Scrollable';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import {
  useActiveProjectId,
  useHasWorkbenchProvider,
  useWorkbenchQueries,
  useWorkbenchSubscription,
} from '@workbench/WorkbenchContext';
import { SearchIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { SettingsSection } from './catalog';

import { browseSettings, isSettingsQuery, settingsCatalog } from './catalog';
import { SettingsBrowseBody, SettingsSearchField } from './SettingsBrowseBody';
import {
  closeWorkbenchSettings,
  getSettingsSectionScroll,
  rememberSettingsSectionScroll,
  setSettingsQuery,
  setWorkbenchSettingsSection,
  settingsDialogStore,
} from './settingsDialogStore';
import { useAvailableSettings } from './useAvailableSettings';

/** Warm code, never mounted editors or data queries, before exposing section navigation. */
export const prepareSettingsDialog = async (): Promise<void> => {
  const resources = new Set(settingsCatalog.flatMap((section) => section.entries.map((entry) => entry.resource)));
  await Promise.allSettled([
    ...[...resources].map((resource) => resource.load()),
    import('./ApplicationSettingField').then((module) => module.prepareApplicationSettings()),
  ]);
};

const GROUPS = ['application', 'project', 'widgets', 'system'] as const;
const DIRECTION = { base: 'column', sm: 'row' } as const;
const SIDEBAR_WIDTH = { base: 'full', sm: '44', md: '56' };
const SIDEBAR_RIGHT_BORDER = { base: '0', sm: '1px' };
const SIDEBAR_BOTTOM_BORDER = { base: '1px', sm: '0' };
const SEARCH_PADDING = { base: '12', sm: '3' };
const MOBILE_DISPLAY = { base: 'block', sm: 'none' };
const DESKTOP_DISPLAY = { base: 'none', sm: 'block' };
const rememberScroll = (event: UIEvent<HTMLDivElement>) => {
  const current = settingsDialogStore.getSnapshot();
  if (!current.query.trim()) {
    rememberSettingsSectionScroll(current.sectionId, event.currentTarget.scrollTop);
  }
};
const clearSearch = () => setSettingsQuery('');
const clearRevealedEntry = () => settingsDialogStore.patchSnapshot({ entryId: undefined });

const ProjectLifetime = () => {
  const projectId = useActiveProjectId();
  const queries = useWorkbenchQueries();
  const subscribe = useWorkbenchSubscription();
  useMountEffect(() =>
    subscribe(() => {
      if (!queries.isActiveProject(projectId)) {
        closeWorkbenchSettings();
      }
    })
  );
  return null;
};

const SettingsDialog = () => {
  const { t } = useTranslation();
  const hasWorkbench = useHasWorkbenchProvider();
  const state = settingsDialogStore.useSelector((snapshot) => snapshot);
  const sections = useAvailableSettings();
  const { active, count, displayed, matches, searching } = browseSettings(
    sections,
    { activeId: state.sectionId, query: state.query, searchSection: state.searchSection },
    t
  );
  const selectSection = useCallback(
    (sectionId: string) => {
      if (isSettingsQuery(state.query)) {
        settingsDialogStore.patchSnapshot({ searchSection: sectionId || null });
      } else {
        setWorkbenchSettingsSection(sectionId);
      }
    },
    [state.query]
  );
  const selectAllResults = useCallback(() => selectSection(''), [selectSection]);
  const changeSection = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => selectSection(event.target.value),
    [selectSection]
  );
  const attachBody = useCallback((element: HTMLDivElement | null) => {
    const current = settingsDialogStore.getSnapshot();
    if (element && !current.query.trim() && !current.entryId) {
      element.scrollTop = getSettingsSectionScroll(current.sectionId);
    }
  }, []);
  const target = useMemo(
    () => (state.target ? { sectionId: state.sectionId, target: state.target } : undefined),
    [state.sectionId, state.target]
  );
  return (
    <Flex h="full" minH="0" direction={DIRECTION}>
      {hasWorkbench ? <ProjectLifetime /> : null}
      <Flex
        as="aside"
        w={SIDEBAR_WIDTH}
        flexShrink={0}
        direction="column"
        bg="bg"
        borderRightWidth={SIDEBAR_RIGHT_BORDER}
        borderBottomWidth={SIDEBAR_BOTTOM_BORDER}
        borderColor="border.subtle"
        minH="0"
      >
        <Box p="3" pe={SEARCH_PADDING}>
          <SettingsSearchField size="sm" value={state.query} onChange={setSettingsQuery} />
        </Box>
        <Box display={MOBILE_DISPLAY} px="3" pb="3">
          <NativeSelect.Root size="sm">
            <NativeSelect.Field
              aria-label={t('settingsDialog.section')}
              value={searching ? (state.searchSection ?? '') : active.id}
              onChange={changeSection}
            >
              {searching ? <option value="">{t('settingsDialog.allResults')}</option> : null}
              {matches.map((section) => (
                <option key={section.id} value={section.id}>
                  {resolveSettingsText(section.label, t)}
                </option>
              ))}
            </NativeSelect.Field>
            <NativeSelect.Indicator />
          </NativeSelect.Root>
        </Box>
        <Scrollable as="nav" aria-label={t('settings.title')} display={DESKTOP_DISPLAY} flex="1" minH="0" px="2" pb="3">
          {searching ? (
            <Button
              w="full"
              justifyContent="space-between"
              size="sm"
              variant={!state.searchSection ? 'subtle' : 'ghost'}
              onClick={selectAllResults}
            >
              {t('settingsDialog.allResults')}
              <Text fontSize="xs">{count}</Text>
            </Button>
          ) : null}
          {GROUPS.map((group) => {
            const groupSections = matches.filter((section) => section.group === group);
            if (!groupSections.length) {
              return null;
            }
            return (
              <Stack key={group} gap="0.5" mt="4">
                <Text px="2" pb="1" fontSize="2xs" fontWeight="600" color="fg.muted" textTransform="uppercase">
                  {t(`settingsDialog.groups.${group}`)}
                </Text>
                {groupSections.map((section) => (
                  <SettingsNavigationItem
                    key={section.id}
                    section={section}
                    selected={(searching ? state.searchSection : active.id) === section.id}
                    searching={searching}
                    onSelect={selectSection}
                  />
                ))}
              </Stack>
            );
          })}
        </Scrollable>
      </Flex>
      <Flex direction="column" flex="1" minW="0" minH="0">
        <Dialog.Header asChild>
          <PanelHeader px="4" pe="12" py="0">
            <HStack gap="2">
              <Icon as={searching ? SearchIcon : active.icon} boxSize="4" />
              <Dialog.Title fontSize="xs" fontWeight="700">
                <VisuallyHidden>{t('settings.title')}: </VisuallyHidden>
                {searching ? t('settingsDialog.results') : resolveSettingsText(active.label, t)}
              </Dialog.Title>
            </HStack>
          </PanelHeader>
        </Dialog.Header>
        <SettingsBrowseBody
          count={count}
          displayed={displayed}
          revealEntryId={state.entryId}
          searching={searching}
          target={target}
          viewKey={searching ? `search:${state.searchSection ?? ''}` : active.id}
          viewportRef={attachBody}
          onClearSearch={clearSearch}
          onReveal={setWorkbenchSettingsSection}
          onRevealed={clearRevealedEntry}
          onScroll={rememberScroll}
        />
      </Flex>
    </Flex>
  );
};

const SettingsNavigationItem = ({
  section,
  selected,
  searching,
  onSelect,
}: {
  section: SettingsSection;
  selected: boolean;
  searching: boolean;
  onSelect: (sectionId: string) => void;
}) => {
  const { t } = useTranslation();
  const select = useCallback(() => onSelect(section.id), [onSelect, section.id]);
  return (
    <Button
      w="full"
      justifyContent="start"
      size="sm"
      variant={selected ? 'subtle' : 'ghost'}
      aria-current={selected ? 'page' : undefined}
      onClick={select}
    >
      <Icon as={section.icon} boxSize="3.5" flexShrink={0} />
      <Text flex="1" textAlign="start" whiteSpace="normal">
        {resolveSettingsText(section.label, t)}
      </Text>
      {searching ? <Text fontSize="2xs">{section.entries.length}</Text> : null}
    </Button>
  );
};

export default SettingsDialog;
