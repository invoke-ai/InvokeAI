import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, Flex, IconButton, Kbd, Text, Tooltip } from '@invoke-ai/ui-library';
import type { AppThunkDispatch } from 'app/store/store';
import type { Hotkey, HotkeyConflictInfo } from 'features/system/components/HotkeysModal/useHotkeyData';
import { IS_MAC_OS } from 'features/system/components/HotkeysModal/useHotkeyData';
import { hotkeyChanged, hotkeyReset } from 'features/system/store/hotkeysSlice';
import type { TFunction } from 'i18next';
import { Fragment, memo, useCallback, useEffect, useMemo, useState } from 'react';
import {
  PiArrowCounterClockwiseBold,
  PiCheckBold,
  PiPencilSimpleBold,
  PiPlusBold,
  PiTrashBold,
  PiXBold,
} from 'react-icons/pi';

// Normalize key names for consistent storage
// Maps platform-specific modifier keys to the cross-platform 'mod' format used by react-hotkeys-hook
// On Mac: Meta (Command) → mod
// On Linux/Windows: Control → mod
const normalizeKey = (key: string): string => {
  const keyMap: Record<string, string> = IS_MAC_OS
    ? {
        Meta: 'mod',
        Command: 'mod',
        Control: 'ctrl',
        Alt: 'alt',
        Shift: 'shift',
        ' ': 'space',
      }
    : {
        Control: 'mod', // On non-Mac, Ctrl is the primary modifier (mapped to 'mod')
        Meta: 'meta', // Windows key - rarely used for hotkeys
        Alt: 'alt',
        Shift: 'shift',
        ' ': 'space',
      };
  return keyMap[key] || key.toLowerCase();
};

// Order of modifiers for consistent output
// 'mod' is the cross-platform primary modifier (Cmd on Mac, Ctrl on Linux/Windows)
// 'ctrl' is only used on Mac (when Ctrl is pressed separately from Cmd)
// 'meta' is only used on non-Mac (Windows key)
const MODIFIER_ORDER = ['mod', 'ctrl', 'meta', 'shift', 'alt'];

const isModifierKey = (key: string): boolean => {
  return ['mod', 'ctrl', 'meta', 'shift', 'alt', 'control', 'command'].includes(key.toLowerCase());
};

// Build hotkey string from pressed keys
const buildHotkeyString = (keys: Set<string>): string | null => {
  const normalizedKeys = Array.from(keys).map(normalizeKey);
  const modifiers = normalizedKeys.filter((k) => MODIFIER_ORDER.includes(k));
  const regularKeys = normalizedKeys.filter((k) => !MODIFIER_ORDER.includes(k));

  // Must have at least one non-modifier key
  if (regularKeys.length === 0) {
    return null;
  }

  // Sort modifiers in consistent order
  const sortedModifiers = modifiers.sort((a, b) => MODIFIER_ORDER.indexOf(a) - MODIFIER_ORDER.indexOf(b));

  // Combine modifiers + regular key (only use first regular key)
  return [...sortedModifiers, regularKeys[0]].join('+');
};

// Format key for display (platform-aware)
const formatKeyForDisplay = (key: string): string => {
  if (IS_MAC_OS) {
    return key.replaceAll('mod', 'cmd').replaceAll('alt', 'option');
  }
  return key.replaceAll('mod', 'ctrl');
};

type HotkeyEditProps = {
  onEditStart?: (index: number) => void;
  onEditCancel: () => void;
  onEditSave: (newHotkey: string, index: number) => void;
  onEditDelete?: (index: number) => void;
};

type HotkeyItemProps = HotkeyEditProps & {
  sx?: SystemStyleObject;
  keyString: string;
  isEditing?: boolean;
  hotkeyIndex: number;
  currentHotkeyId: string;
  isNewHotkey?: boolean;
  conflictMap: Map<string, HotkeyConflictInfo>;
  t: TFunction;
};

const HotkeyRecorderSx: SystemStyleObject = {
  px: 2,
  py: 1,
  bg: 'base.700',
  borderRadius: 'base',
  borderWidth: 1,
  borderColor: 'invokeBlue.400',
  minW: '100px',
  justifyContent: 'center',
  alignItems: 'center',
  cursor: 'pointer',
};

const HotkeyRecorderConflictSx: SystemStyleObject = {
  ...HotkeyRecorderSx,
  borderColor: 'error.400',
  bg: 'error.900',
};

const HotkeyItem = memo(
  ({
    hotkeyIndex,
    keyString,
    sx,
    isEditing,
    onEditCancel,
    onEditSave,
    onEditStart,
    onEditDelete,
    currentHotkeyId,
    isNewHotkey,
    conflictMap,
    t,
  }: HotkeyItemProps) => {
    const [recordedKey, setRecordedKey] = useState<string | null>(null);
    const [isRecording, setIsRecording] = useState(false);

    // Memoize key parts to avoid repeated split calls
    const keyParts = useMemo(() => keyString.split('+'), [keyString]);
    const displayKeyParts = useMemo(() => keyParts.map(formatKeyForDisplay), [keyParts]);

    // Check if the recorded key conflicts with another hotkey
    const conflict = useMemo(() => {
      if (!recordedKey) {
        return null;
      }
      const existingHotkey = conflictMap.get(recordedKey);
      if (!existingHotkey) {
        return null;
      }
      // Don't flag conflict if it's the same hotkey we're editing
      if (existingHotkey.fullId === currentHotkeyId) {
        return null;
      }
      return existingHotkey;
    }, [recordedKey, conflictMap, currentHotkeyId]);

    // Start recording when entering edit mode
    useEffect(() => {
      if (isEditing) {
        setRecordedKey(null);
        setIsRecording(true);
      } else {
        setIsRecording(false);
        setRecordedKey(null);
      }
    }, [isEditing]);

    // Handle keyboard events during recording
    useEffect(() => {
      if (!isRecording) {
        return;
      }

      const handleKeyDown = (e: globalThis.KeyboardEvent) => {
        e.preventDefault();
        e.stopPropagation();

        // Escape cancels editing
        if (e.key === 'Escape') {
          onEditCancel();
          return;
        }

        // Ignore pure modifier key presses
        if (isModifierKey(e.key)) {
          return;
        }

        // Build the complete key combination
        const keys = new Set<string>();
        if (e.ctrlKey) {
          keys.add('Control');
        }
        if (e.shiftKey) {
          keys.add('Shift');
        }
        if (e.altKey) {
          keys.add('Alt');
        }
        if (e.metaKey) {
          keys.add('Meta');
        }
        keys.add(e.key);

        const hotkeyString = buildHotkeyString(keys);
        if (hotkeyString) {
          setRecordedKey(hotkeyString);
          setIsRecording(false);
        }
      };

      const handleKeyUp = (e: globalThis.KeyboardEvent) => {
        e.preventDefault();
        e.stopPropagation();
      };

      window.addEventListener('keydown', handleKeyDown, true);
      window.addEventListener('keyup', handleKeyUp, true);

      return () => {
        window.removeEventListener('keydown', handleKeyDown, true);
        window.removeEventListener('keyup', handleKeyUp, true);
      };
    }, [isRecording, onEditCancel]);

    const onCancelEdit = useCallback(() => {
      setRecordedKey(null);
      onEditCancel();
    }, [onEditCancel]);

    const onStartEdit = useCallback(() => {
      onEditStart?.(hotkeyIndex);
    }, [onEditStart, hotkeyIndex]);

    const onSaveEdit = useCallback(() => {
      // Use recorded key, or fall back to original normalized key string
      const keyToSave = recordedKey ?? keyString;
      onEditSave(keyToSave, hotkeyIndex);
    }, [onEditSave, recordedKey, keyString, hotkeyIndex]);

    const onDeleteEdit = useCallback(() => {
      onEditDelete?.(hotkeyIndex);
    }, [onEditDelete, hotkeyIndex]);

    const startRecording = useCallback(() => {
      setIsRecording(true);
      setRecordedKey(null);
    }, []);

    const canSaveEdit = useMemo(() => {
      // Cannot save if no key recorded
      if (!recordedKey) {
        return false;
      }
      // Cannot save if there is a conflict
      if (conflict) {
        return false;
      }
      return true;
    }, [recordedKey, conflict]);

    // Render the hotkey display or editor
    const renderHotkeyKeys = () => {
      if (isEditing) {
        const displayKey = recordedKey ?? keyString;
        const parts = displayKey.split('+').map(formatKeyForDisplay);
        const hasConflict = conflict !== null;

        return (
          <Tooltip
            isOpen={hasConflict}
            label={hasConflict ? t('hotkeys.conflictWarning', { hotkeyTitle: conflict?.title }) : undefined}
            bg="error.900"
            color="error.100"
          >
            <Flex sx={hasConflict ? HotkeyRecorderConflictSx : HotkeyRecorderSx} onClick={startRecording}>
              {isRecording ? (
                <Text fontSize="xs" color="invokeBlue.300" fontStyle="italic">
                  {t('hotkeys.pressKeys')}
                </Text>
              ) : displayKey ? (
                parts.map((part, j) => (
                  <Fragment key={j}>
                    <Kbd fontSize="xs" textTransform="lowercase">
                      {part}
                    </Kbd>
                    {j !== parts.length - 1 && (
                      <Text as="span" fontSize="xs" fontWeight="semibold" mx={0.5}>
                        +
                      </Text>
                    )}
                  </Fragment>
                ))
              ) : (
                <Text fontSize="xs" color="invokeBlue.300" fontStyle="italic">
                  {t('hotkeys.pressKeys')}
                </Text>
              )}
            </Flex>
          </Tooltip>
        );
      }

      return (
        <Tooltip label={t('hotkeys.editHotkey')}>
          <Button
            variant="ghost"
            size="sm"
            onClick={onStartEdit}
            rightIcon={<PiPencilSimpleBold />}
            gap={0.5}
            alignItems="center"
            px={2}
          >
            {displayKeyParts.map((part, j) => (
              <Fragment key={j}>
                <Kbd fontSize="xs" textTransform="lowercase">
                  {part}
                </Kbd>
                {j !== displayKeyParts.length - 1 && (
                  <Text as="span" fontSize="xs" fontWeight="semibold" mx={0.5} mt={-0.5}>
                    +
                  </Text>
                )}
              </Fragment>
            ))}
          </Button>
        </Tooltip>
      );
    };

    return (
      <Flex sx={sx}>
        {renderHotkeyKeys()}
        <Flex gap={1}>
          {isEditing && (
            <>
              {!isNewHotkey && (
                <Tooltip label={t('common.delete')}>
                  <IconButton
                    aria-label={t('common.delete')}
                    icon={<PiTrashBold />}
                    size="sm"
                    variant="ghost"
                    colorScheme="error"
                    onClick={onDeleteEdit}
                  />
                </Tooltip>
              )}
              <Tooltip label={t('hotkeys.cancel')}>
                <IconButton
                  aria-label={t('hotkeys.cancel')}
                  icon={<PiXBold />}
                  size="sm"
                  variant="ghost"
                  onClick={onCancelEdit}
                />
              </Tooltip>
              <Tooltip label={t('hotkeys.save')}>
                <IconButton
                  aria-label={t('hotkeys.save')}
                  icon={<PiCheckBold />}
                  size="sm"
                  variant="ghost"
                  onClick={onSaveEdit}
                  disabled={!canSaveEdit}
                />
              </Tooltip>
            </>
          )}
        </Flex>
      </Flex>
    );
  }
);

HotkeyItem.displayName = 'HotkeyItem';

type HotkeyItemsDisplayProps = HotkeyEditProps & {
  sx?: SystemStyleObject;
  hotkeys: string[]; // Original normalized hotkeys (not platform-formatted)
  editingIndex?: number | null;
  onAddHotkey?: () => void;
  currentHotkeyId: string;
  conflictMap: Map<string, HotkeyConflictInfo>;
  isCustomized?: boolean;
  onReset?: () => void;
  t: TFunction;
};

const HotkeyItemsDisplaySx: SystemStyleObject = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-end',
  gap: 1,
};

const HotkeyItemsDisplay = memo(
  ({
    hotkeys,
    sx,
    editingIndex,
    onEditStart,
    onEditCancel,
    onEditSave,
    onEditDelete,
    onAddHotkey,
    currentHotkeyId,
    conflictMap,
    isCustomized,
    onReset,
    t,
  }: HotkeyItemsDisplayProps) => {
    const isAddingNew = editingIndex === hotkeys.length;

    return (
      <Flex sx={{ ...HotkeyItemsDisplaySx, ...sx }}>
        {hotkeys.length > 0
          ? hotkeys.map((keyString, i) => (
              <HotkeyItem
                key={i}
                hotkeyIndex={i}
                keyString={keyString}
                isEditing={editingIndex === i}
                onEditStart={onEditStart}
                onEditCancel={onEditCancel}
                onEditSave={onEditSave}
                onEditDelete={onEditDelete}
                currentHotkeyId={currentHotkeyId}
                conflictMap={conflictMap}
                t={t}
              />
            ))
          : !isAddingNew && (
              <Text fontSize="sm" color="base.400">
                {t('hotkeys.noHotkeysRecorded')}
              </Text>
            )}
        {isAddingNew && (
          <HotkeyItem
            hotkeyIndex={hotkeys.length}
            keyString=""
            isEditing={true}
            onEditStart={onEditStart}
            onEditCancel={onEditCancel}
            onEditSave={onEditSave}
            onEditDelete={onEditDelete}
            currentHotkeyId={currentHotkeyId}
            isNewHotkey={true}
            conflictMap={conflictMap}
            t={t}
          />
        )}
        <Flex>
          {isCustomized && (
            <Tooltip label={t('hotkeys.resetToDefault')}>
              <IconButton
                aria-label={t('hotkeys.resetToDefault')}
                icon={<PiArrowCounterClockwiseBold />}
                size="sm"
                variant="ghost"
                colorScheme="warning"
                onClick={onReset}
              />
            </Tooltip>
          )}
          {!isAddingNew && (
            <Tooltip label={t('hotkeys.addHotkey')}>
              <IconButton
                aria-label={t('hotkeys.addHotkey')}
                icon={<PiPlusBold />}
                variant="ghost"
                size="sm"
                onClick={onAddHotkey}
              />
            </Tooltip>
          )}
        </Flex>
      </Flex>
    );
  }
);

HotkeyItemsDisplay.displayName = 'HotkeyItemsDisplay';

type HotkeyListItemProps = {
  lastItem?: boolean;
  hotkey: Hotkey;
  sx?: SystemStyleObject;
  conflictMap: Map<string, HotkeyConflictInfo>;
  t: TFunction;
  dispatch: AppThunkDispatch;
};

const HotkeyListItemSx: SystemStyleObject = {
  alignItems: 'start',
  justifyContent: 'space-between',
  width: '100%',
  py: 3,
  gap: 2,
};

export const HotkeyListItem = memo(({ lastItem, hotkey, sx, conflictMap, t, dispatch }: HotkeyListItemProps) => {
  const { title, desc, hotkeys: hotkeyKeys, defaultHotkeys } = hotkey;

  const [editingIndex, setEditingIndex] = useState<number | null>(null);

  // Check if hotkeys have been customized
  const isCustomized = useMemo(() => {
    if (hotkeyKeys.length !== defaultHotkeys.length) {
      return true;
    }
    return hotkeyKeys.some((key, i) => key !== defaultHotkeys[i]);
  }, [hotkeyKeys, defaultHotkeys]);

  const currentHotkeyId = `${hotkey.category}.${hotkey.id}`;

  const handleStartEdit = useCallback((index: number) => {
    setEditingIndex(index);
  }, []);

  const handleCancel = useCallback(() => {
    setEditingIndex(null);
  }, []);

  const handleSave = useCallback(
    (newHotkey: string, index: number) => {
      // Skip saving empty hotkeys
      if (!newHotkey) {
        setEditingIndex(null);
        return;
      }

      // Skip saving hotkey if it already exists in the list
      if (hotkeyKeys.includes(newHotkey)) {
        setEditingIndex(null);
        return;
      }

      const updatedHotkeys = [...hotkeyKeys];
      if (index >= updatedHotkeys.length) {
        // Adding a new hotkey
        updatedHotkeys.push(newHotkey);
      } else {
        // Updating an existing hotkey
        updatedHotkeys[index] = newHotkey;
      }

      dispatch(hotkeyChanged({ id: currentHotkeyId, hotkeys: updatedHotkeys }));
      setEditingIndex(null);
    },
    [dispatch, currentHotkeyId, hotkeyKeys]
  );

  const handleAddHotkey = useCallback(() => {
    // Set editing index to the next available slot
    setEditingIndex(hotkeyKeys.length);
  }, [hotkeyKeys.length]);

  const handleDelete = useCallback(
    (index: number) => {
      const updatedHotkeys = hotkeyKeys.filter((_, i) => i !== index);
      dispatch(hotkeyChanged({ id: currentHotkeyId, hotkeys: updatedHotkeys }));
      setEditingIndex(null);
    },
    [dispatch, currentHotkeyId, hotkeyKeys]
  );

  const handleReset = useCallback(() => {
    dispatch(hotkeyReset(currentHotkeyId));
  }, [dispatch, currentHotkeyId]);

  return (
    <Flex sx={{ ...HotkeyListItemSx, borderBottomWidth: lastItem ? 0 : 1, ...sx }}>
      <Flex lineHeight={1} gap={2} w="100%" flexDir="column">
        <Text fontWeight="semibold">{title}</Text>
        <Text variant="subtext">{desc}</Text>
      </Flex>
      <Flex gap={2} w="100%" justifyContent="flex-end">
        <HotkeyItemsDisplay
          hotkeys={hotkeyKeys}
          editingIndex={editingIndex}
          onEditStart={handleStartEdit}
          onEditCancel={handleCancel}
          onEditSave={handleSave}
          onEditDelete={handleDelete}
          onAddHotkey={handleAddHotkey}
          currentHotkeyId={currentHotkeyId}
          conflictMap={conflictMap}
          isCustomized={isCustomized}
          onReset={handleReset}
          t={t}
        />
      </Flex>
    </Flex>
  );
});

HotkeyListItem.displayName = 'HotkeyListItem';
