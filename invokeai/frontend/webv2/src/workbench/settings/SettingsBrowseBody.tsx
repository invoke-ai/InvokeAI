import type { SettingsTarget } from '@platform/ui/settings/contracts';
import type { ChangeEvent, Ref, UIEventHandler } from 'react';

import { Box, HStack, Icon, Input, InputGroup, Kbd, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { SearchIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { SettingsSection } from './catalog';

import { isFillSettingsEntry, SettingsEntryView } from './SettingsEntryView';
import { SettingsScopeLabel } from './SettingsScopeLabel';
import { patchWorkbenchPreferences, useWorkbenchSettingsSelector } from './store';

const CONTENT_PADDING = { base: '4', md: '6' };
const FILL_SECTION_PROPS = { display: 'flex', flex: '1', flexDirection: 'column', minH: '0' } as const;
const FILL_BODY_PROPS = { ...FILL_SECTION_PROPS, pb: '4' } as const;
const BODY_CONTENT_PROPS = { px: CONTENT_PADDING, pb: '4' } as const;
const retrySave = () => {
  void patchWorkbenchPreferences({});
};
const SEARCH_START_ELEMENT = <Icon as={SearchIcon} boxSize="3.5" color="fg.muted" />;
/** The `/` shortcut its surface handles with `focusSettingsSearchOnSlash`. */
const SEARCH_HOTKEY_HINT = (
  <Kbd aria-hidden pointerEvents="none" size="sm" variant="outline">
    /
  </Kbd>
);

export const SettingsSearchField = ({
  onChange,
  size,
  value,
}: {
  onChange: (query: string) => void;
  size: 'xs' | 'sm';
  value: string;
}) => {
  const { t } = useTranslation();
  const change = useCallback((event: ChangeEvent<HTMLInputElement>) => onChange(event.target.value), [onChange]);
  const clear = useCallback(() => onChange(''), [onChange]);
  // The end slot is a clear control while there is a query, and the `/` hint otherwise.
  const endElement = useMemo(
    () =>
      value ? (
        <Button aria-label={t('settingsDialog.clearSearch')} me="-2" size="2xs" variant="ghost" onClick={clear}>
          <XIcon />
        </Button>
      ) : (
        SEARCH_HOTKEY_HINT
      ),
    [clear, t, value]
  );
  return (
    <InputGroup endElement={endElement} startElement={SEARCH_START_ELEMENT}>
      <Input
        data-settings-search
        aria-label={t('settingsDialog.search')}
        placeholder={t('settingsDialog.search')}
        value={value}
        size={size}
        onChange={change}
      />
    </InputGroup>
  );
};

export interface SettingsBrowseBodyProps {
  displayed: readonly SettingsSection[];
  searching: boolean;
  /** Matching entries across every section while searching. */
  count: number;
  /** Changes when the shown content changes, so the scroll region starts fresh. */
  viewKey: string;
  /** A widget instance the settings were opened for, and the section it belongs to. */
  target?: { sectionId: string; target: SettingsTarget };
  revealEntryId?: string;
  onRevealed?: () => void;
  onReveal: (sectionId: string, entryId?: string) => void;
  onClearSearch: () => void;
  onScroll?: UIEventHandler<HTMLDivElement>;
  viewportRef?: Ref<HTMLDivElement>;
}

/** The part of a settings surface below its title: save errors, the shown sections, and search results. */
export const SettingsBrowseBody = ({
  count,
  displayed,
  onClearSearch,
  onReveal,
  onRevealed,
  onScroll,
  revealEntryId,
  searching,
  target,
  viewKey,
  viewportRef,
}: SettingsBrowseBodyProps) => {
  const { t } = useTranslation();
  const error = useWorkbenchSettingsSelector((snapshot) => snapshot.error);
  const viewportProps = useMemo(() => ({ onScroll }), [onScroll]);
  const sectionProps = { onReveal, onRevealed, revealEntryId };
  // A filling editor (the hotkeys table) scrolls itself: the body hands it the
  // height instead of wrapping it in a second scroll container.
  const fills =
    !searching && displayed.length === 1 && displayed[0].entries.some((entry) => isFillSettingsEntry(entry, 'dialog'));
  return (
    <>
      {error ? (
        <HStack role="alert" px="4" py="2" bg="bg.error">
          <Text fontSize="xs" color="fg.error" flex="1">
            {error}
          </Text>
          <Button size="xs" onClick={retrySave}>
            {t('common.retry')}
          </Button>
        </HStack>
      ) : null}
      <VisuallyHidden role="status">{searching ? t('settingsDialog.resultCount', { count }) : ''}</VisuallyHidden>
      {fills ? (
        <Box key={viewKey} px={CONTENT_PADDING} {...FILL_BODY_PROPS}>
          {displayed.map((section) => (
            <SettingsSectionContent
              key={section.id}
              fill
              section={section}
              search={false}
              target={target?.sectionId === section.id ? target.target : undefined}
              {...sectionProps}
            />
          ))}
        </Box>
      ) : (
        <Scrollable
          key={viewKey}
          contentProps={BODY_CONTENT_PROPS}
          flex="1"
          minH="0"
          viewportProps={viewportProps}
          viewportRef={viewportRef}
        >
          {displayed.map((section) => (
            <SettingsSectionContent
              key={section.id}
              section={section}
              search={searching}
              target={target?.sectionId === section.id ? target.target : undefined}
              {...sectionProps}
            />
          ))}
          {searching && !displayed.length ? (
            <Stack align="center" py="12" gap="3">
              <Text color="fg.muted">{t('settingsDialog.noResults')}</Text>
              <Button size="sm" variant="outline" onClick={onClearSearch}>
                {t('settingsDialog.clearSearch')}
              </Button>
            </Stack>
          ) : null}
        </Scrollable>
      )}
    </>
  );
};

const SettingsSectionContent = ({
  fill = false,
  section,
  search,
  target,
  onReveal,
  revealEntryId,
  onRevealed,
}: {
  fill?: boolean;
  section: SettingsSection;
  search: boolean;
  target?: SettingsTarget;
  onReveal: (sectionId: string, entryId?: string) => void;
  revealEntryId?: string;
  onRevealed?: () => void;
}) => {
  const { t } = useTranslation();
  return (
    <Box {...(fill ? FILL_SECTION_PROPS : undefined)}>
      {!search && section.entries[0] ? (
        <Box pt="3">
          <SettingsScopeLabel scope={section.entries[0].field.scope} />
        </Box>
      ) : null}
      {search ? (
        <Stack gap="0.5" pt="5">
          <Text as="h3" fontWeight="600" fontSize="sm">
            {resolveSettingsText(section.label, t)}
          </Text>
          {section.entries[0] ? <SettingsScopeLabel scope={section.entries[0].field.scope} /> : null}
        </Stack>
      ) : null}
      {section.entries.map((entry, index) => (
        <SettingsEntryView
          key={entry.field.id}
          entry={entry}
          section={section}
          target={target}
          search={search}
          revealEntryId={revealEntryId}
          onReveal={onReveal}
          onRevealed={onRevealed}
          showGroup={Boolean(
            entry.field.group &&
            resolveSettingsText(entry.field.group, t) !==
              (section.entries[index - 1]?.field.group
                ? resolveSettingsText(section.entries[index - 1].field.group!, t)
                : '')
          )}
        />
      ))}
    </Box>
  );
};
