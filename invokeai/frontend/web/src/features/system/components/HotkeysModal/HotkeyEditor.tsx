import { Flex, IconButton, Input, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { Hotkey } from 'features/system/components/HotkeysModal/useHotkeyData';
import { hotkeyChanged, hotkeyReset } from 'features/system/store/hotkeysSlice';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiCheckBold, PiPencilBold, PiXBold } from 'react-icons/pi';

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

  const handleSave = useCallback(() => {
    const newKeys = parseHotkeyInput(editValue);
    if (newKeys.length > 0) {
      const hotkeyId = `${hotkey.category}.${hotkey.id}`;
      dispatch(hotkeyChanged({ id: hotkeyId, hotkeys: newKeys }));
    }
    setIsEditing(false);
    setEditValue('');
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

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  if (isEditing) {
    return (
      <Flex gap={2} alignItems="center">
        <Input
          ref={inputRef}
          value={editValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={t('hotkeys.enterHotkeys')}
          size="sm"
        />
        <IconButton
          aria-label={t('hotkeys.save')}
          icon={<PiCheckBold />}
          onClick={handleSave}
          size="sm"
          colorScheme="invokeBlue"
        />
        <IconButton aria-label={t('hotkeys.cancel')} icon={<PiXBold />} onClick={handleCancel} size="sm" />
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
