import {
  Box,
  Flex,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Tooltip,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbaColorPicker from 'common/components/ColorPicker/RgbaColorPicker';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import {
  selectCanvasSettingsSlice,
  settingsActiveColorToggled,
  settingsColor1Changed,
  settingsColor2Changed,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selectActiveColor = createSelector(selectCanvasSettingsSlice, (settings) => settings.activeColor);
const selectColor1 = createSelector(selectCanvasSettingsSlice, (settings) => settings.color1);
const selectColor2 = createSelector(selectCanvasSettingsSlice, (settings) => settings.color2);

export const ToolFillColorPicker = memo(() => {
  const { t } = useTranslation();
  const activeColorType = useAppSelector(selectActiveColor);
  const color1 = useAppSelector(selectColor1);
  const color2 = useAppSelector(selectColor2);
  const { activeColor, tooltip, color1zIndex, color2zIndex } = useMemo(() => {
    if (activeColorType === 'color1') {
      return { activeColor: color1, tooltip: t('controlLayers.fill.fillColor1'), color1zIndex: 2, color2zIndex: 1 };
    } else {
      return { activeColor: color2, tooltip: t('controlLayers.fill.fillColor2'), color1zIndex: 1, color2zIndex: 2 };
    }
  }, [activeColorType, color1, color2, t]);
  const dispatch = useAppDispatch();
  const onColorChange = useCallback(
    (color: RgbaColor) => {
      if (activeColorType === 'color1') {
        dispatch(settingsColor1Changed(color));
      } else {
        dispatch(settingsColor2Changed(color));
      }
    },
    [activeColorType, dispatch]
  );

  useRegisteredHotkeys({
    id: 'toggleFillColor',
    category: 'canvas',
    callback: () => dispatch(settingsActiveColorToggled()),
    options: { preventDefault: true },
    dependencies: [dispatch],
  });

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Flex role="button" aria-label={t('controlLayers.fill.fillColor')} tabIndex={-1} w={8} h={8}>
          <Tooltip label={tooltip}>
            <Flex alignItems="center" justifyContent="center" position="relative" w="full" h="full">
              <Box
                borderRadius="full"
                borderColor="base.600"
                w={6}
                h={6}
                borderWidth={2}
                bg={rgbaColorToString(color1)}
                position="absolute"
                top="0"
                left="0"
                zIndex={color1zIndex}
              />
              <Box
                borderRadius="full"
                borderColor="base.600"
                w={6}
                h={6}
                borderWidth={2}
                bg={rgbaColorToString(color2)}
                position="absolute"
                top="2"
                left="2"
                zIndex={color2zIndex}
              />
            </Flex>
          </Tooltip>
        </Flex>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody minH={64}>
          <RgbaColorPicker color={activeColor} onChange={onColorChange} withNumberInput withSwatches />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

ToolFillColorPicker.displayName = 'ToolFillColorPicker';
