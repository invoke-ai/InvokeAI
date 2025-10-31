import { Button, Flex, IconButton, Input, Kbd, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { Hotkey } from 'features/system/components/HotkeysModal/useHotkeyData';
import { hotkeyChanged, hotkeyReset } from 'features/system/store/hotkeysSlice';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { Fragment, memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiCheckBold, PiInfoBold, PiPencilBold, PiXBold } from 'react-icons/pi';

type HotkeyEditorProps = {
  hotkey: Hotkey;
};

const formatHotkeyForDisplay = (keys: string[]): string => {
  return keys.join(', ');
};

const parseHotkeyInput = (input: string): string[] => {
  return input
    .split(',')
    .map((k) => k.trim())
    .filter((k) => k.length > 0);
};

const validateHotkey = (hotkey: string): boolean => {
  // Basic validation: check if hotkey has valid format
  const parts = hotkey.split('+').map((p) => p.trim());
  if (parts.length === 0) {
    return false;
  }
  // Last part should be a key, not a modifier
  const lastPart = parts[parts.length - 1];
  if (!lastPart) {
    return false;
  }
  return lastPart.length > 0 && !['mod', 'ctrl', 'shift', 'alt', 'meta'].includes(lastPart.toLowerCase());
};

export const HotkeyEditor = memo(({ hotkey }: HotkeyEditorProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const isCustomized = hotkey.hotkeys.join(',') !== hotkey.defaultHotkeys.join(',');

  const handleEdit = useCallback(() => {
    setEditValue(formatHotkeyForDisplay(hotkey.hotkeys));
    setIsEditing(true);
  }, [hotkey.hotkeys]);

  const handleCancel = useCallback(() => {
    setIsEditing(false);
    setEditValue('');
  }, []);

  const isValid = useMemo(() => {
    const keys = parseHotkeyInput(editValue);
    return keys.length > 0 && keys.every((k) => validateHotkey(k));
  }, [editValue]);

  const previewKeys = useMemo(() => {
    return parseHotkeyInput(editValue);
  }, [editValue]);

  const handleSave = useCallback(() => {
    const newKeys = parseHotkeyInput(editValue);
    if (newKeys.length > 0 && newKeys.every((k) => validateHotkey(k))) {
      const hotkeyId = `${hotkey.category}.${hotkey.id}`;
      dispatch(hotkeyChanged({ id: hotkeyId, hotkeys: newKeys }));
      setIsEditing(false);
      setEditValue('');
    }
  }, [dispatch, editValue, hotkey.category, hotkey.id]);

  const handleReset = useCallback(() => {
    const hotkeyId = `${hotkey.category}.${hotkey.id}`;
    dispatch(hotkeyReset(hotkeyId));
  }, [dispatch, hotkey.category, hotkey.id]);

  const handleInputChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setEditValue(e.target.value);
  }, []);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSave();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        handleCancel();
      }
    },
    [handleSave, handleCancel]
  );

  // Insert modifier at cursor position
  const insertModifier = useCallback(
    (modifier: string) => {
      if (!inputRef.current) {
        return;
      }

      const input = inputRef.current;
      const start = input.selectionStart ?? editValue.length;
      const end = input.selectionEnd ?? editValue.length;
      const before = editValue.slice(0, start);
      const after = editValue.slice(end);

      // Smart insertion: add + if not at start and previous char is not a separator
      const needsPrefix = start > 0 && before.slice(-1) !== '+' && before.slice(-1) !== ',' && before.slice(-1) !== ' ';
      const prefix = needsPrefix ? '+' : '';

      const newValue = `${before + prefix + modifier}+${after}`;
      setEditValue(newValue);

      // Move cursor after the inserted modifier and +
      setTimeout(() => {
        const newPosition = start + prefix.length + modifier.length + 1;
        input.setSelectionRange(newPosition, newPosition);
        input.focus();
      }, 0);
    },
    [editValue]
  );

  const insertMod = useCallback(() => insertModifier('mod'), [insertModifier]);
  const insertCtrl = useCallback(() => insertModifier('ctrl'), [insertModifier]);
  const insertShift = useCallback(() => insertModifier('shift'), [insertModifier]);
  const insertAlt = useCallback(() => insertModifier('alt'), [insertModifier]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  if (isEditing) {
    return (
      <Flex direction="column" gap={2} w="full">
        <Flex gap={2} alignItems="center">
          <Input
            ref={inputRef}
            value={editValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={t('hotkeys.enterHotkeys')}
            size="sm"
            flex={1}
            borderColor={editValue && !isValid ? 'error.500' : undefined}
          />
          <IconButton
            aria-label={t('hotkeys.save')}
            icon={<PiCheckBold />}
            onClick={handleSave}
            size="sm"
            colorScheme="invokeBlue"
            isDisabled={!isValid}
          />
          <IconButton aria-label={t('hotkeys.cancel')} icon={<PiXBold />} onClick={handleCancel} size="sm" />
        </Flex>
        <Flex gap={2} alignItems="center" flexWrap="wrap">
          <Text fontSize="xs" color="base.300">
            {t('hotkeys.modifiers')}:
          </Text>
          <Button size="xs" onClick={insertMod} variant="outline">
            Mod
          </Button>
          <Button size="xs" onClick={insertCtrl} variant="outline">
            Ctrl
          </Button>
          <Button size="xs" onClick={insertShift} variant="outline">
            Shift
          </Button>
          <Button size="xs" onClick={insertAlt} variant="outline">
            Alt
          </Button>
          <Tooltip
            label={
              <Flex direction="column" gap={1} fontSize="xs">
                <Text fontWeight="semibold">{t('hotkeys.syntaxHelp')}</Text>
                <Text>• mod = Ctrl (Win/Linux) / Cmd (Mac)</Text>
                <Text>• {t('hotkeys.combineWith')}: mod+shift+a</Text>
                <Text>• {t('hotkeys.multipleHotkeys')}: mod+a, ctrl+b</Text>
                <Text>• {t('hotkeys.validKeys')}: a-z, 0-9, f1-f12, enter, space, etc.</Text>
              </Flex>
            }
            placement="bottom-start"
          >
            <IconButton
              aria-label={t('hotkeys.help')}
              icon={<PiInfoBold />}
              size="xs"
              variant="ghost"
              colorScheme="base"
            />
          </Tooltip>
        </Flex>
        {previewKeys.length > 0 && (
          <Flex gap={2} alignItems="center" flexWrap="wrap" opacity={isValid ? 1 : 0.5}>
            <Text fontSize="xs" color="base.300">
              Preview:
            </Text>
            {previewKeys.map((key, i) => (
              <Fragment key={i}>
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
                {i !== previewKeys.length - 1 && (
                  <Text as="span" fontSize="xs" px={1} variant="subtext">
                    {t('common.or')}
                  </Text>
                )}
              </Fragment>
            ))}
          </Flex>
        )}
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
