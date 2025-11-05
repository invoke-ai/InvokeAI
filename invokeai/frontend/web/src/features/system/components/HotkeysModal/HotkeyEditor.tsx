import { Button, Flex, IconButton, Kbd, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { Hotkey } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useHotkeyData } from 'features/system/components/HotkeysModal/useHotkeyData';
import { hotkeyChanged, hotkeyReset } from 'features/system/store/hotkeysSlice';
import { Fragment, memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiCheckBold, PiPencilBold, PiTrashBold, PiXBold } from 'react-icons/pi';

type HotkeyEditorProps = {
  hotkey: Hotkey;
};

const formatHotkeyForDisplay = (keys: string[]): string => {
  return keys.join(', ');
};

// Normalize key names for consistent storage
const normalizeKey = (key: string): string => {
  const keyMap: Record<string, string> = {
    Control: 'ctrl',
    Meta: 'mod',
    Command: 'mod',
    Alt: 'alt',
    Shift: 'shift',
    ' ': 'space',
  };
  return keyMap[key] || key.toLowerCase();
};

// Order of modifiers for consistent output
const MODIFIER_ORDER = ['mod', 'ctrl', 'shift', 'alt'];

const isModifierKey = (key: string): boolean => {
  return ['mod', 'ctrl', 'shift', 'alt', 'control', 'meta', 'command'].includes(key.toLowerCase());
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

export const HotkeyEditor = memo(({ hotkey }: HotkeyEditorProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const allHotkeysData = useHotkeyData();
  const [isEditing, setIsEditing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedHotkeys, setRecordedHotkeys] = useState<string[]>([]);
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());
  const [duplicateWarning, setDuplicateWarning] = useState<{ hotkeyString: string; conflictTitle: string } | null>(
    null
  );

  const isCustomized = hotkey.hotkeys.join(',') !== hotkey.defaultHotkeys.join(',');

  // Build a flat map of all hotkeys for conflict detection
  const allHotkeysMap = useMemo(() => {
    const map = new Map<string, { category: string; id: string; title: string }>();
    Object.entries(allHotkeysData).forEach(([category, categoryData]) => {
      Object.entries(categoryData.hotkeys).forEach(([id, hotkeyData]) => {
        hotkeyData.hotkeys.forEach((hotkeyString) => {
          map.set(hotkeyString, { category, id, title: hotkeyData.title });
        });
      });
    });
    return map;
  }, [allHotkeysData]);

  // Check if a hotkey conflicts with another hotkey (not the current one)
  const findConflict = useCallback(
    (hotkeyString: string): { title: string } | null => {
      const conflict = allHotkeysMap.get(hotkeyString);
      if (!conflict) {
        return null;
      }
      // Check if it's the same hotkey we're editing
      const currentHotkeyId = `${hotkey.category}.${hotkey.id}`;
      const conflictId = `${conflict.category}.${conflict.id}`;
      if (currentHotkeyId === conflictId) {
        // It's the same hotkey, check if it's already in recordedHotkeys
        return null;
      }
      return { title: conflict.title };
    },
    [allHotkeysMap, hotkey.category, hotkey.id]
  );

  const handleEdit = useCallback(() => {
    setRecordedHotkeys([...hotkey.hotkeys]);
    setIsEditing(true);
    setIsRecording(false);
  }, [hotkey.hotkeys]);

  const handleCancel = useCallback(() => {
    setIsEditing(false);
    setIsRecording(false);
    setRecordedHotkeys([]);
    setPressedKeys(new Set());
  }, []);

  const handleSave = useCallback(() => {
    if (recordedHotkeys.length > 0) {
      const hotkeyId = `${hotkey.category}.${hotkey.id}`;
      dispatch(hotkeyChanged({ id: hotkeyId, hotkeys: recordedHotkeys }));
      setIsEditing(false);
      setIsRecording(false);
      setRecordedHotkeys([]);
      setPressedKeys(new Set());
    }
  }, [dispatch, recordedHotkeys, hotkey.category, hotkey.id]);

  const handleReset = useCallback(() => {
    const hotkeyId = `${hotkey.category}.${hotkey.id}`;
    dispatch(hotkeyReset(hotkeyId));
  }, [dispatch, hotkey.category, hotkey.id]);

  const startRecording = useCallback(() => {
    setIsRecording(true);
    setPressedKeys(new Set());
    setDuplicateWarning(null);
  }, []);

  const clearLastRecorded = useCallback(() => {
    setRecordedHotkeys((prev) => prev.slice(0, -1));
  }, []);

  const clearAllRecorded = useCallback(() => {
    setRecordedHotkeys([]);
  }, []);

  // Handle keyboard events during recording
  useEffect(() => {
    if (!isRecording) {
      return;
    }

    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();

      // Ignore pure modifier keys being pressed
      if (isModifierKey(e.key)) {
        setPressedKeys((prev) => new Set(prev).add(e.key));
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

      setPressedKeys(keys);

      // Build hotkey string
      const hotkeyString = buildHotkeyString(keys);
      if (hotkeyString) {
        // Check for duplicates in current recorded hotkeys
        setRecordedHotkeys((prev) => {
          if (prev.includes(hotkeyString)) {
            setDuplicateWarning({ hotkeyString, conflictTitle: t('hotkeys.thisHotkey') });
            setIsRecording(false);
            setPressedKeys(new Set());
            return prev;
          }

          // Check for conflicts with other hotkeys in the system
          const conflict = findConflict(hotkeyString);
          if (conflict) {
            setDuplicateWarning({ hotkeyString, conflictTitle: conflict.title });
            setIsRecording(false);
            setPressedKeys(new Set());
            return prev;
          }

          setDuplicateWarning(null);
          setIsRecording(false);
          setPressedKeys(new Set());
          return [...prev, hotkeyString];
        });
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
  }, [isRecording, findConflict, t]);

  if (isEditing) {
    return (
      <Flex direction="column" gap={3} w="full">
        {/* Recorded hotkeys display */}
        <Flex direction="column" gap={2}>
          <Flex gap={2} alignItems="center" flexWrap="wrap" minH={8}>
            {recordedHotkeys.length > 0 ? (
              <>
                {recordedHotkeys.map((key, i) => (
                  <Fragment key={i}>
                    <Flex gap={1} alignItems="center">
                      {key.split('+').map((part, j) => (
                        <Fragment key={j}>
                          <Kbd fontSize="xs" textTransform="lowercase">
                            {part}
                          </Kbd>
                          {j !== key.split('+').length - 1 && (
                            <Text as="span" fontSize="xs" fontWeight="semibold">
                              +
                            </Text>
                          )}
                        </Fragment>
                      ))}
                    </Flex>
                    {i !== recordedHotkeys.length - 1 && (
                      <Text as="span" fontSize="xs" px={1} variant="subtext">
                        {t('common.or')}
                      </Text>
                    )}
                  </Fragment>
                ))}
              </>
            ) : (
              <Text fontSize="sm" color="base.400">
                {t('hotkeys.noHotkeysRecorded')}
              </Text>
            )}
          </Flex>
        </Flex>

        {/* Recording indicator */}
        {isRecording && (
          <Flex
            gap={2}
            alignItems="center"
            p={3}
            bg="invokeBlue.500"
            borderRadius="md"
            borderWidth={2}
            borderColor="invokeBlue.300"
          >
            <Text fontSize="sm" fontWeight="bold" color="white">
              {t('hotkeys.pressKeys')}
            </Text>
            {pressedKeys.size > 0 && (
              <Flex gap={1}>
                {Array.from(pressedKeys).map((key) => (
                  <Kbd key={key} fontSize="xs" bg="white" color="black">
                    {normalizeKey(key)}
                  </Kbd>
                ))}
              </Flex>
            )}
          </Flex>
        )}

        {/* Duplicate/Conflict warning */}
        {duplicateWarning && (
          <Flex gap={2} alignItems="center" p={2} bg="error.500" borderRadius="md" opacity={0.9}>
            <Text fontSize="xs" color="white">
              <Kbd fontSize="xs" bg="white" color="black">
                {duplicateWarning.hotkeyString}
              </Kbd>{' '}
              {t('hotkeys.conflictWarning', { hotkeyTitle: duplicateWarning.conflictTitle })}
            </Text>
          </Flex>
        )}

        {/* Action buttons */}
        <Flex gap={2} flexWrap="wrap">
          {!isRecording && (
            <>
              <Button size="sm" onClick={startRecording} colorScheme="invokeBlue">
                {recordedHotkeys.length > 0 ? t('hotkeys.setAnother') : t('hotkeys.setHotkey')}
              </Button>
              {recordedHotkeys.length > 0 && (
                <>
                  <Tooltip label={t('hotkeys.removeLastHotkey')}>
                    <IconButton
                      aria-label={t('hotkeys.removeLastHotkey')}
                      icon={<PiTrashBold />}
                      onClick={clearLastRecorded}
                      size="sm"
                      variant="ghost"
                    />
                  </Tooltip>
                  <Button size="sm" onClick={clearAllRecorded} variant="ghost">
                    {t('hotkeys.clearAll')}
                  </Button>
                </>
              )}
            </>
          )}
        </Flex>

        {/* Save/Cancel buttons */}
        <Flex gap={2}>
          <Button
            size="sm"
            onClick={handleSave}
            colorScheme="invokeBlue"
            leftIcon={<PiCheckBold />}
            isDisabled={recordedHotkeys.length === 0 || isRecording}
            flex={1}
          >
            {t('hotkeys.save')}
          </Button>
          <Button size="sm" onClick={handleCancel} variant="ghost" leftIcon={<PiXBold />}>
            {t('hotkeys.cancel')}
          </Button>
        </Flex>
      </Flex>
    );
  }

  return (
    <Flex gap={2} alignItems="center">
      <Text fontSize="sm" fontWeight="semibold" minW={32}>
        {formatHotkeyForDisplay(hotkey.hotkeys)}
      </Text>
      <Tooltip label={t('hotkeys.editHotkey')}>
        <IconButton
          aria-label={t('hotkeys.editHotkey')}
          icon={<PiPencilBold />}
          onClick={handleEdit}
          size="sm"
          variant="ghost"
        />
      </Tooltip>
      {isCustomized && (
        <Tooltip label={t('hotkeys.resetToDefault')}>
          <IconButton
            aria-label={t('hotkeys.resetToDefault')}
            icon={<PiArrowCounterClockwiseBold />}
            onClick={handleReset}
            size="sm"
            variant="ghost"
          />
        </Tooltip>
      )}
    </Flex>
  );
});

HotkeyEditor.displayName = 'HotkeyEditor';
