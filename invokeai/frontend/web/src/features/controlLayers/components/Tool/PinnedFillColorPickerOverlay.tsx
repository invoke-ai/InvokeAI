import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbaColorPicker from 'common/components/ColorPicker/RgbaColorPicker';
import {
  selectCanvasSettingsSlice,
  settingsActiveColorToggled,
  settingsBgColorChanged,
  settingsFgColorChanged,
  settingsFillColorPickerPinnedSet,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsLeftRightBold, PiPushPinSlashBold } from 'react-icons/pi';

const selectActiveColor = createSelector(selectCanvasSettingsSlice, (settings) => settings.activeColor);
const selectBgColor = createSelector(selectCanvasSettingsSlice, (settings) => settings.bgColor);
const selectFgColor = createSelector(selectCanvasSettingsSlice, (settings) => settings.fgColor);
const selectPinned = createSelector(selectCanvasSettingsSlice, (settings) => settings.fillColorPickerPinned);

export const PinnedFillColorPickerOverlay = memo(() => {
  const { t } = useTranslation();
  const isPinned = useAppSelector(selectPinned);
  const activeColorType = useAppSelector(selectActiveColor);
  const bgColor = useAppSelector(selectBgColor);
  const fgColor = useAppSelector(selectFgColor);
  const dispatch = useAppDispatch();

  const activeColor = useMemo(
    () => (activeColorType === 'bgColor' ? bgColor : fgColor),
    [activeColorType, bgColor, fgColor]
  );

  const onColorChange = useCallback(
    (color: RgbaColor) => {
      if (activeColorType === 'bgColor') {
        dispatch(settingsBgColorChanged(color));
      } else {
        dispatch(settingsFgColorChanged(color));
      }
    },
    [activeColorType, dispatch]
  );

  const onUnpin = useCallback(() => dispatch(settingsFillColorPickerPinnedSet(false)), [dispatch]);
  const onToggleActive = useCallback(() => dispatch(settingsActiveColorToggled()), [dispatch]);

  if (!isPinned) {
    return null;
  }

  return (
    <Flex pointerEvents="auto" direction="column" gap={2}>
      <Flex
        direction="column"
        p={3}
        bg="base.900"
        borderColor="base.700"
        borderWidth="1px"
        borderStyle="solid"
        shadow="dark-lg"
        borderRadius="base"
        minW={88}
      >
        <Flex justifyContent="space-between" alignItems="center" mb={2} gap={2}>
          <Text fontWeight="semibold" color="base.300">
            {t('controlLayers.fill.fillColor')}
          </Text>
          <Flex gap={1}>
            <IconButton
              aria-label={t('controlLayers.fill.switchColors', { defaultValue: 'Switch FG/BG (X)' })}
              tooltip={t('controlLayers.fill.switchColors', { defaultValue: 'Switch FG/BG (X)' })}
              size="sm"
              variant="ghost"
              onClick={onToggleActive}
              icon={<PiArrowsLeftRightBold />}
            />
            <IconButton
              aria-label={t('common.unpin', { defaultValue: 'Unpin' })}
              tooltip={t('common.unpin', { defaultValue: 'Unpin' })}
              size="sm"
              variant="solid"
              onClick={onUnpin}
              icon={<PiPushPinSlashBold />}
            />
          </Flex>
        </Flex>
        <RgbaColorPicker color={activeColor} onChange={onColorChange} withNumberInput withSwatches />
      </Flex>
    </Flex>
  );
});

PinnedFillColorPickerOverlay.displayName = 'PinnedFillColorPickerOverlay';
