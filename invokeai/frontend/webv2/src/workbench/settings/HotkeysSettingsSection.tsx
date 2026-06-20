import type { HotkeyCategory, HotkeyDefinition } from '@workbench/hotkeys';

import { Badge, Box, Flex, HStack, Icon, Input, InputGroup, Kbd, ScrollArea, Stack, Text } from '@chakra-ui/react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Button, IconButton } from '@workbench/components/ui';
import {
  firstPartyHotkeyCatalog,
  formatHotkeyForPlatform,
  normalizeHotkeyString,
  eventToHotkeyString,
  useExtensionHotkeyDefinitions,
} from '@workbench/hotkeys';
import { patchWorkbenchPreferences, useWorkbenchPreferences } from '@workbench/settings/store';
import { CheckIcon, PlusIcon, RotateCcwIcon, SearchIcon, Trash2Icon, XIcon } from 'lucide-react';
import { Fragment, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';

type HotkeyConflict = { hotkey: HotkeyDefinition; title: string };

type HotkeyRow =
  | { kind: 'category'; category: HotkeyCategory; title: string }
  | { kind: 'hotkey'; hotkey: HotkeyDefinition; effectiveKeys: string[]; isCustomized: boolean };

const CATEGORY_LABELS: Record<HotkeyCategory, string> = {
  app: 'App',
  canvas: 'Canvas',
  gallery: 'Gallery',
  viewer: 'Viewer',
  workflows: 'Workflows',
};

const CATEGORY_ORDER: HotkeyCategory[] = ['app', 'canvas', 'workflows', 'viewer', 'gallery'];

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

const buildRows = ({
  catalog,
  customHotkeys,
  searchTerm,
}: {
  catalog: HotkeyDefinition[];
  customHotkeys: Record<string, string[]>;
  searchTerm: string;
}): HotkeyRow[] => {
  const needle = searchTerm.trim().toLowerCase();
  const rows: HotkeyRow[] = [];

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

    if (hotkeys.length === 0) {
      continue;
    }

    rows.push({ category, kind: 'category', title: CATEGORY_LABELS[category] });

    for (const hotkey of hotkeys) {
      rows.push({
        effectiveKeys: normalizeKeys(customHotkeys[hotkey.id] ?? hotkey.defaultKeys),
        hotkey,
        isCustomized: isHotkeyCustomized(hotkey, customHotkeys),
        kind: 'hotkey',
      });
    }
  }

  return rows;
};

export const HotkeysSettingsSection = () => {
  const { customHotkeys } = useWorkbenchPreferences();
  const extensionHotkeys = useExtensionHotkeyDefinitions();
  const [searchTerm, setSearchTerm] = useState('');
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const catalog = useMemo(() => [...firstPartyHotkeyCatalog, ...extensionHotkeys], [extensionHotkeys]);
  const rows = useMemo(
    () => buildRows({ catalog, customHotkeys, searchTerm: deferredSearchTerm }),
    [catalog, customHotkeys, deferredSearchTerm]
  );
  const conflictMap = useMemo(() => buildConflictMap(catalog, customHotkeys), [catalog, customHotkeys]);
  const modifiedCount = Object.keys(customHotkeys).length;
  const virtualizer = useVirtualizer({
    count: rows.length,
    estimateSize: (index) => (rows[index]?.kind === 'category' ? 36 : 86),
    getScrollElement: () => scrollRef.current,
    measureElement: (element) => element.getBoundingClientRect().height,
    overscan: 8,
  });

  const saveHotkey = (hotkeyId: string, keys: string[]) => {
    void patchWorkbenchPreferences({ customHotkeys: { ...customHotkeys, [hotkeyId]: normalizeKeys(keys) } });
  };

  const resetHotkey = (hotkeyId: string) => {
    const next = { ...customHotkeys };

    delete next[hotkeyId];
    void patchWorkbenchPreferences({ customHotkeys: next });
  };

  const resetAll = () => {
    void patchWorkbenchPreferences({ customHotkeys: {} });
  };

  const virtualRows = virtualizer.getVirtualItems();

  return (
    <Stack h="full" minH="0">
      <HStack justify="space-between" gap="3">
        <Stack gap="0.5">
          <Text color="fg" fontSize="sm" fontWeight="600">
            Hotkeys
          </Text>
          <Text color="fg.subtle" fontSize="xs">
            Account-bound keybinds. Project-level overrides can layer on later without changing hotkey ids.
          </Text>
        </Stack>
        <Button disabled={modifiedCount === 0} size="xs" variant="outline" onClick={resetAll}>
          <RotateCcwIcon />
          Reset all
        </Button>
      </HStack>

      <InputGroup startElement={<Icon as={SearchIcon} boxSize="3.5" />}>
        <Input
          placeholder="Search hotkeys"
          size="xs"
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.currentTarget.value)}
        />
      </InputGroup>

      <ScrollArea.Root flex="1" minH="0" rounded="md" size="xs" variant="hover" w="full">
        <ScrollArea.Viewport ref={scrollRef} aria-label="Hotkey bindings" h="full" w="full">
          <ScrollArea.Content w="full">
            <Box h={`${virtualizer.getTotalSize()}px`} position="relative" w="full">
              {virtualRows.map((row) => {
                const item = rows[row.index];

                if (!item) {
                  return null;
                }

                return (
                  <Box
                    key={row.key}
                    ref={virtualizer.measureElement}
                    data-index={row.index}
                    position="absolute"
                    top="0"
                    transform={`translateY(${row.start}px)`}
                    w="full"
                  >
                    {item.kind === 'category' ? (
                      <CategoryRow title={item.title} />
                    ) : (
                      <HotkeyListRow
                        conflictMap={conflictMap}
                        effectiveKeys={item.effectiveKeys}
                        hotkey={item.hotkey}
                        isCustomized={item.isCustomized}
                        onReset={() => resetHotkey(item.hotkey.id)}
                        onSave={(keys) => saveHotkey(item.hotkey.id, keys)}
                      />
                    )}
                  </Box>
                );
              })}
            </Box>
          </ScrollArea.Content>
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar>
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
    </Stack>
  );
};

const CategoryRow = ({ title }: { title: string }) => (
  <Flex align="center" borderBottomWidth="1px" pb="2" pt="4">
    <Text color="fg.muted" fontSize="2xs" fontWeight="800" letterSpacing="0.08em" textTransform="uppercase">
      {title}
    </Text>
  </Flex>
);

const HotkeyListRow = ({
  conflictMap,
  effectiveKeys,
  hotkey,
  isCustomized,
  onReset,
  onSave,
}: {
  conflictMap: Map<string, HotkeyConflict[]>;
  effectiveKeys: string[];
  hotkey: HotkeyDefinition;
  isCustomized: boolean;
  onReset: () => void;
  onSave: (keys: string[]) => void;
}) => {
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

  useEffect(() => {
    if (!isEditing) {
      setDraftKeys(effectiveKeys);
    }
  }, [effectiveKeys, isEditing]);

  const updateDraftKey = (index: number, key: string) => {
    setDraftKeys((current) => current.map((candidate, candidateIndex) => (candidateIndex === index ? key : candidate)));
  };

  const deleteDraftKey = (index: number) => {
    setDraftKeys((current) => current.filter((_key, candidateIndex) => candidateIndex !== index));
    setEditingIndex(null);
  };

  const addDraftKey = () => {
    setDraftKeys((current) => [...current, '']);
    setEditingIndex(draftKeys.length);
  };

  const cancelEdit = () => {
    setDraftKeys(effectiveKeys);
    setEditingIndex(null);
  };

  const saveEdit = () => {
    if (!canSave) {
      return;
    }

    onSave(draftKeys);
    setEditingIndex(null);
  };

  const disableHotkey = () => {
    onSave([]);
    setEditingIndex(null);
  };

  return (
    <Flex borderBottomWidth="1px" gap="3" py="2.5">
      <Stack flex="1" gap="1" minW="0">
        <HStack gap="2">
          <Text color="fg" fontSize="sm" fontWeight="600" truncate>
            {getHotkeyTitle(hotkey)}
          </Text>
          {hotkey.implemented === false ? (
            <Badge colorPalette="gray" size="xs" variant="surface">
              Pending
            </Badge>
          ) : null}
          {isCustomized ? (
            <Badge colorPalette="blue" size="xs" variant="surface">
              Custom
            </Badge>
          ) : null}
        </HStack>
        <Text color="fg.subtle" fontSize="2xs" truncate>
          {hotkey.description ?? hotkey.id}
        </Text>
        {conflict ? (
          <Text color="fg.error" fontSize="2xs">
            Conflicts with {conflict.title}
          </Text>
        ) : hasDuplicate ? (
          <Text color="fg.error" fontSize="2xs">
            Duplicate binding in this hotkey
          </Text>
        ) : null}
      </Stack>
      <Stack align="end" gap="1.5" maxW="58%">
        <HStack gap="1.5" justify="end" wrap="wrap">
          {draftKeys.length > 0 ? (
            draftKeys.map((key, index) => (
              <HotkeyChip
                key={`${index}:${key}`}
                editing={editingIndex === index}
                hotkey={key}
                onCancel={() => setEditingIndex(null)}
                onDelete={() => deleteDraftKey(index)}
                onEdit={() => setEditingIndex(index)}
                onRecord={(nextKey) => updateDraftKey(index, nextKey)}
              />
            ))
          ) : (
            <Text color="fg.subtle" fontSize="2xs">
              Disabled
            </Text>
          )}
          <IconButton aria-label="Add hotkey" size="xs" variant="ghost" onClick={addDraftKey}>
            <PlusIcon />
          </IconButton>
        </HStack>
        <HStack gap="1">
          {isEditing || isDirty ? (
            <>
              <IconButton aria-label="Cancel hotkey edit" size="xs" variant="ghost" onClick={cancelEdit}>
                <XIcon />
              </IconButton>
              <IconButton
                aria-label="Save hotkey edit"
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
              Disable
            </Button>
          ) : null}
          {isCustomized ? (
            <IconButton aria-label="Reset hotkey" size="xs" variant="ghost" onClick={onReset}>
              <RotateCcwIcon />
            </IconButton>
          ) : null}
        </HStack>
      </Stack>
    </Flex>
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
          Press keys
        </Text>
        <IconButton aria-label="Delete hotkey" colorPalette="red" size="2xs" variant="ghost" onClick={onDelete}>
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
            {part}
          </Kbd>
          {index < parts.length - 1 ? (
            <Text color="fg.subtle" fontSize="2xs">
              +
            </Text>
          ) : null}
        </Fragment>
      ))}
    </Button>
  );
};
