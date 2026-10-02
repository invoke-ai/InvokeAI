import type { ListRowProps } from '@platform/ui/list/List';
import type { ListSection } from '@platform/ui/list/listRows';
import type { HotkeyCategory, HotkeyDefinition } from '@workbench/hotkeys';

import { Badge, HStack, Icon, Input, InputGroup, Kbd, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { List } from '@platform/ui/list/List';
import { listRowsFromSections } from '@platform/ui/list/listRows';
import { ModifiedSettingIndicator } from '@platform/ui/settings/ModifiedSettingIndicator';
import {
  firstPartyHotkeyCatalog,
  formatHotkeyForPlatform,
  normalizeHotkeyString,
  eventToHotkeyString,
  useExtensionHotkeyDefinitions,
} from '@workbench/hotkeys';
import { ShortcutKeyGlyph } from '@workbench/hotkeys/keyGlyphs';
import { patchWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { CheckIcon, PlusIcon, RotateCcwIcon, SearchIcon, Trash2Icon, XIcon } from 'lucide-react';
import { Fragment, useCallback, useDeferredValue, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

type HotkeyConflict = { hotkey: HotkeyDefinition; title: string };

interface HotkeyEntry {
  hotkey: HotkeyDefinition;
  effectiveKeys: string[];
  isCustomized: boolean;
}

const CATEGORY_LABEL_KEYS: Record<HotkeyCategory, string> = {
  app: 'hotkeys.categories.app',
  canvas: 'hotkeys.categories.canvas',
  gallery: 'hotkeys.categories.gallery',
  viewer: 'hotkeys.categories.viewer',
  workflows: 'hotkeys.categories.workflows',
};

const CATEGORY_ORDER: HotkeyCategory[] = ['app', 'canvas', 'workflows', 'viewer', 'gallery'];

const SEARCH_START_ELEMENT = <Icon as={SearchIcon} boxSize="3.5" />;
const NO_MATCHES_ICON = <Icon as={SearchIcon} />;

const getEntryKey = (entry: HotkeyEntry): string => entry.hotkey.id;

const normalizeKeys = (keys: string[]): string[] => keys.map(normalizeHotkeyString).filter(Boolean);

const titleFromId = (id: string): string =>
  id
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replaceAll('.', ' ')
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

const getHotkeyTitle = (hotkey: HotkeyDefinition): string => hotkey.title || titleFromId(hotkey.id);

const getScopeRank = (hotkey: HotkeyDefinition): number => {
  if (hotkey.scope.kind === 'instance') {
    return 400;
  }

  if (hotkey.scope.kind === 'widget') {
    return 300;
  }

  if (hotkey.scope.kind === 'focused-region') {
    return 200;
  }

  return 100;
};

const canScopesOverlap = (left: HotkeyDefinition, right: HotkeyDefinition): boolean => {
  if (getScopeRank(left) !== getScopeRank(right)) {
    return false;
  }

  if (left.scope.kind === 'global' && right.scope.kind === 'global') {
    return true;
  }

  if (left.scope.kind === 'widget' && right.scope.kind === 'widget') {
    return left.scope.typeId === right.scope.typeId;
  }

  if (left.scope.kind === 'instance' && right.scope.kind === 'instance') {
    return left.scope.instanceId === right.scope.instanceId;
  }

  if (left.scope.kind === 'focused-region' && right.scope.kind === 'focused-region') {
    return !left.scope.region || !right.scope.region || left.scope.region === right.scope.region;
  }

  return false;
};

const buildConflictMap = (
  catalog: HotkeyDefinition[],
  customHotkeys: Record<string, string[]>
): Map<string, HotkeyConflict[]> => {
  const conflicts = new Map<string, HotkeyConflict[]>();

  for (const hotkey of catalog) {
    if (hotkey.implemented === false) {
      continue;
    }

    const effectiveKeys = normalizeKeys(customHotkeys[hotkey.id] ?? hotkey.defaultKeys);

    for (const key of effectiveKeys) {
      const entries = conflicts.get(key) ?? [];

      entries.push({ hotkey, title: getHotkeyTitle(hotkey) });
      conflicts.set(key, entries);
    }
  }

  return conflicts;
};

const isHotkeyCustomized = (hotkey: HotkeyDefinition, customHotkeys: Record<string, string[]>): boolean =>
  Object.prototype.hasOwnProperty.call(customHotkeys, hotkey.id);

const buildSections = ({
  catalog,
  customHotkeys,
  searchTerm,
  t,
}: {
  catalog: HotkeyDefinition[];
  customHotkeys: Record<string, string[]>;
  searchTerm: string;
  t: (key: string) => string;
}): ListSection<HotkeyEntry>[] => {
  const needle = searchTerm.trim().toLowerCase();
  const sections: ListSection<HotkeyEntry>[] = [];

  for (const category of CATEGORY_ORDER) {
    const hotkeys = catalog
      .filter((hotkey) => hotkey.category === category)
      .filter((hotkey) => {
        if (!needle) {
          return true;
        }

        const haystack = [getHotkeyTitle(hotkey), hotkey.description, hotkey.id, hotkey.defaultKeys.join(' ')]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();

        return haystack.includes(needle);
      });

    sections.push({
      items: hotkeys.map((hotkey) => ({
        effectiveKeys: normalizeKeys(customHotkeys[hotkey.id] ?? hotkey.defaultKeys),
        hotkey,
        isCustomized: isHotkeyCustomized(hotkey, customHotkeys),
      })),
      key: category,
      label: t(CATEGORY_LABEL_KEYS[category]),
    });
  }

  return sections;
};

export const HotkeysSettingsSection = () => {
  const { t } = useTranslation();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const extensionHotkeys = useExtensionHotkeyDefinitions();
  const [searchTerm, setSearchTerm] = useState('');
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const catalog = useMemo(() => [...firstPartyHotkeyCatalog, ...extensionHotkeys], [extensionHotkeys]);
  const rows = useMemo(
    () =>
      listRowsFromSections(buildSections({ catalog, customHotkeys, searchTerm: deferredSearchTerm, t }), getEntryKey),
    [catalog, customHotkeys, deferredSearchTerm, t]
  );
  const conflictMap = useMemo(() => buildConflictMap(catalog, customHotkeys), [catalog, customHotkeys]);
  const modifiedCount = Object.keys(customHotkeys).length;

  const saveHotkey = useCallback(
    (hotkeyId: string, keys: string[]) => {
      void patchWorkbenchPreferences({ customHotkeys: { ...customHotkeys, [hotkeyId]: normalizeKeys(keys) } });
    },
    [customHotkeys]
  );

  const resetHotkey = useCallback(
    (hotkeyId: string) => {
      const next = { ...customHotkeys };

      delete next[hotkeyId];
      void patchWorkbenchPreferences({ customHotkeys: next });
    },
    [customHotkeys]
  );

  const resetAll = useCallback(() => {
    void patchWorkbenchPreferences({ customHotkeys: {} });
  }, []);
  const handleSearchChange = useCallback((event: { currentTarget: { value: string } }) => {
    setSearchTerm(event.currentTarget.value);
  }, []);

  const emptyState = useMemo(() => <EmptyState icon={NO_MATCHES_ICON} title={t('hotkeys.noMatches')} />, [t]);
  const renderItem = useCallback(
    (entry: HotkeyEntry, rowProps: ListRowProps) => (
      <HotkeyListRow
        key={entry.effectiveKeys.join('\n')}
        conflictMap={conflictMap}
        effectiveKeys={entry.effectiveKeys}
        hotkey={entry.hotkey}
        isCustomized={entry.isCustomized}
        positionInSet={rowProps.positionInSet}
        setSize={rowProps.setSize}
        onReset={resetHotkey}
        onSave={saveHotkey}
      />
    ),
    [conflictMap, resetHotkey, saveHotkey]
  );

  return (
    <Stack h="full" minH="0">
      <HStack justify="space-between" gap="3">
        <Stack gap="0.5">
          <Text color="fg" fontSize="sm" fontWeight="600">
            {t('hotkeys.title')}
          </Text>
          <Text color="fg.muted" fontSize="xs">
            {t('hotkeys.description')}
          </Text>
        </Stack>
        <Button disabled={modifiedCount === 0} size="xs" variant="outline" onClick={resetAll}>
          <RotateCcwIcon />
          {t('hotkeys.resetAll')}
        </Button>
      </HStack>

      <InputGroup startElement={SEARCH_START_ELEMENT}>
        <Input
          aria-label={t('hotkeys.searchPlaceholder')}
          placeholder={t('hotkeys.searchPlaceholder')}
          size="xs"
          value={searchTerm}
          onChange={handleSearchChange}
        />
      </InputGroup>

      <List
        density="comfortable"
        dividers
        emptyState={emptyState}
        estimatedRowHeight={64}
        label={t('hotkeys.bindings')}
        renderItem={renderItem}
        rowHeight="measured"
        rows={rows}
        // The Settings dialog body paints bg.subtle; pinned category headers must match it.
        surface="bg.subtle"
      />
    </Stack>
  );
};

const HotkeyListRow = ({
  conflictMap,
  effectiveKeys,
  hotkey,
  isCustomized,
  onReset,
  onSave,
  positionInSet,
  setSize,
}: {
  conflictMap: Map<string, HotkeyConflict[]>;
  effectiveKeys: string[];
  hotkey: HotkeyDefinition;
  isCustomized: boolean;
  onReset: (hotkeyId: string) => void;
  onSave: (hotkeyId: string, keys: string[]) => void;
  positionInSet: number;
  setSize: number;
}) => {
  const { t } = useTranslation();
  const [draftKeys, setDraftKeys] = useState(effectiveKeys);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const isEditing = editingIndex !== null;
  const isDirty = draftKeys.join('\n') !== effectiveKeys.join('\n');
  const hasDuplicate = new Set(draftKeys).size !== draftKeys.length;
  const conflict =
    hotkey.implemented === false
      ? undefined
      : draftKeys
          .flatMap((key) => conflictMap.get(key) ?? [])
          .find((entry) => entry.hotkey.id !== hotkey.id && canScopesOverlap(hotkey, entry.hotkey));
  const canSave = !hasDuplicate && !conflict && (isDirty || isEditing);
  const defaultKeys = normalizeKeys(hotkey.defaultKeys);
  const isModified =
    effectiveKeys.length !== defaultKeys.length || effectiveKeys.some((key) => !defaultKeys.includes(key));

  const updateDraftKey = useCallback((index: number, key: string) => {
    setDraftKeys((current) => current.map((candidate, candidateIndex) => (candidateIndex === index ? key : candidate)));
  }, []);

  const deleteDraftKey = useCallback((index: number) => {
    setDraftKeys((current) => current.filter((_key, candidateIndex) => candidateIndex !== index));
    setEditingIndex(null);
  }, []);

  const addDraftKey = useCallback(() => {
    setDraftKeys((current) => [...current, '']);
    setEditingIndex(draftKeys.length);
  }, [draftKeys.length]);

  const cancelEdit = useCallback(() => {
    setDraftKeys(effectiveKeys);
    setEditingIndex(null);
  }, [effectiveKeys]);

  const saveEdit = useCallback(() => {
    if (!canSave) {
      return;
    }

    onSave(hotkey.id, draftKeys);
    setEditingIndex(null);
  }, [canSave, draftKeys, hotkey.id, onSave]);

  const disableHotkey = useCallback(() => {
    onSave(hotkey.id, []);
    setEditingIndex(null);
  }, [hotkey.id, onSave]);
  const resetThisHotkey = useCallback(() => onReset(hotkey.id), [hotkey.id, onReset]);

  const cancelChipEdit = useCallback(() => setEditingIndex(null), []);

  return (
    // An inline editor, not a ListItem: its controls keep their own tab order inside the list.
    <HStack
      align="flex-start"
      aria-posinset={positionInSet}
      aria-setsize={setSize}
      gap="3"
      px="2"
      py="2"
      role="listitem"
    >
      <Stack flex="1" gap="1" minW="0">
        <HStack gap="2">
          <Text color="fg" fontSize="sm" fontWeight="600" truncate>
            {getHotkeyTitle(hotkey)}
          </Text>
          {hotkey.implemented === false ? (
            <Badge colorPalette="gray" size="xs" variant="surface">
              {t('hotkeys.pending')}
            </Badge>
          ) : null}
          {isModified ? <ModifiedSettingIndicator label={getHotkeyTitle(hotkey)} /> : null}
        </HStack>
        <Text color="fg.muted" fontSize="2xs" truncate>
          {hotkey.description ?? hotkey.id}
        </Text>
        {conflict ? (
          <Text color="fg.error" fontSize="2xs">
            {t('hotkeys.conflictsWith', { title: conflict.title })}
          </Text>
        ) : hasDuplicate ? (
          <Text color="fg.error" fontSize="2xs">
            {t('hotkeys.duplicateBinding')}
          </Text>
        ) : null}
      </Stack>
      <Stack align="end" gap="1.5" maxW="58%">
        <HStack gap="1.5" justify="end" wrap="wrap">
          {draftKeys.length > 0 ? (
            draftKeys.map((key, index) => (
              <HotkeyChipItem
                key={`${index}:${key}`}
                cancelChipEdit={cancelChipEdit}
                deleteDraftKey={deleteDraftKey}
                editingIndex={editingIndex}
                hotkey={key}
                index={index}
                setEditingIndex={setEditingIndex}
                updateDraftKey={updateDraftKey}
              />
            ))
          ) : (
            <Text color="fg.muted" fontSize="2xs">
              {t('hotkeys.disabled')}
            </Text>
          )}
          <IconButton aria-label={t('hotkeys.addHotkey')} size="xs" variant="ghost" onClick={addDraftKey}>
            <PlusIcon />
          </IconButton>
        </HStack>
        <HStack gap="1">
          {isEditing || isDirty ? (
            <>
              <IconButton aria-label={t('hotkeys.cancelEdit')} size="xs" variant="ghost" onClick={cancelEdit}>
                <XIcon />
              </IconButton>
              <IconButton
                aria-label={t('hotkeys.saveEdit')}
                disabled={!canSave}
                size="xs"
                variant="ghost"
                onClick={saveEdit}
              >
                <CheckIcon />
              </IconButton>
            </>
          ) : null}
          {effectiveKeys.length > 0 ? (
            <Button size="xs" variant="ghost" onClick={disableHotkey}>
              {t('hotkeys.disable')}
            </Button>
          ) : null}
          {isCustomized ? (
            <IconButton aria-label={t('hotkeys.resetHotkey')} size="xs" variant="ghost" onClick={resetThisHotkey}>
              <RotateCcwIcon />
            </IconButton>
          ) : null}
        </HStack>
      </Stack>
    </HStack>
  );
};

const HotkeyChip = ({
  editing,
  hotkey,
  onCancel,
  onDelete,
  onEdit,
  onRecord,
}: {
  editing: boolean;
  hotkey: string;
  onCancel: () => void;
  onDelete: () => void;
  onEdit: () => void;
  onRecord: (hotkey: string) => void;
}) => {
  const { t } = useTranslation();
  useEffect(() => {
    if (!editing) {
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();

      if (event.key === 'Escape') {
        onCancel();
        return;
      }

      const nextHotkey = eventToHotkeyString(event);

      if (nextHotkey) {
        onRecord(nextHotkey);
        onCancel();
      }
    };
    const blockKeyUp = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();
    };

    window.addEventListener('keydown', onKeyDown, true);
    window.addEventListener('keyup', blockKeyUp, true);

    return () => {
      window.removeEventListener('keydown', onKeyDown, true);
      window.removeEventListener('keyup', blockKeyUp, true);
    };
  }, [editing, onCancel, onRecord]);

  if (editing) {
    return (
      <HStack borderColor="accent.solid" borderWidth="1px" gap="1" px="2" py="1" rounded="md">
        <Text color="accent.solid" fontSize="2xs" fontStyle="italic">
          {t('hotkeys.pressKeys')}
        </Text>
        <IconButton
          aria-label={t('hotkeys.deleteHotkey')}
          colorPalette="red"
          size="2xs"
          variant="ghost"
          onClick={onDelete}
        >
          <Trash2Icon />
        </IconButton>
      </HStack>
    );
  }

  return (
    <Button type="button" onClick={onEdit} variant="outline" size="xs">
      {formatHotkeyForPlatform(hotkey).map((part, index, parts) => (
        <Fragment key={`${part}:${index}`}>
          <Kbd size="sm" textTransform="lowercase">
            <ShortcutKeyGlyph fallback={part} part={part} />
          </Kbd>
          {index < parts.length - 1 ? (
            <Text color="fg.muted" fontSize="2xs">
              +
            </Text>
          ) : null}
        </Fragment>
      ))}
    </Button>
  );
};

const HotkeyChipItem = ({
  cancelChipEdit,
  deleteDraftKey,
  editingIndex,
  hotkey,
  index,
  setEditingIndex,
  updateDraftKey,
}: {
  cancelChipEdit: () => void;
  deleteDraftKey: (index: number) => void;
  editingIndex: number | null;
  hotkey: string;
  index: number;
  setEditingIndex: (index: number) => void;
  updateDraftKey: (index: number, key: string) => void;
}) => {
  const handleDelete = useCallback(() => deleteDraftKey(index), [deleteDraftKey, index]);
  const handleEdit = useCallback(() => setEditingIndex(index), [index, setEditingIndex]);
  const handleRecord = useCallback((nextKey: string) => updateDraftKey(index, nextKey), [index, updateDraftKey]);

  return (
    <HotkeyChip
      editing={editingIndex === index}
      hotkey={hotkey}
      onCancel={cancelChipEdit}
      onDelete={handleDelete}
      onEdit={handleEdit}
      onRecord={handleRecord}
    />
  );
};
