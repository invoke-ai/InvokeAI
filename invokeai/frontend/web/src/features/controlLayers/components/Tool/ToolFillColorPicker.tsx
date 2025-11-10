import {
  Box,
  Flex,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbaColorPicker from 'common/components/ColorPicker/RgbaColorPicker';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import {
  selectActiveColor,
  selectBgColor,
  selectFgColor,
  settingsActiveColorToggled,
  settingsBgColorChanged,
  settingsColorsSetToDefault,
  settingsFgColorChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ToolFillColorPicker = memo(() => {
  const { t } = useTranslation();
  const activeColorType = useAppSelector((state) => selectActiveColor(state));
  const bgColor = useAppSelector((state) => selectBgColor(state));
  const fgColor = useAppSelector((state) => selectFgColor(state));
  const { activeColor, tooltip, bgColorzIndex, fgColorzIndex } = useMemo(() => {
    if (activeColorType === 'bgColor') {
      return { activeColor: bgColor, tooltip: t('controlLayers.fill.bgFillColor'), bgColorzIndex: 2, fgColorzIndex: 1 };
    } else {
      return { activeColor: fgColor, tooltip: t('controlLayers.fill.fgFillColor'), bgColorzIndex: 1, fgColorzIndex: 2 };
    }
  }, [activeColorType, bgColor, fgColor, t]);
  const dispatch = useAppDispatch();
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

  useRegisteredHotkeys({
    id: 'setFillColorsToDefault',
    category: 'canvas',
    callback: () => dispatch(settingsColorsSetToDefault()),
    options: { preventDefault: true },
    dependencies: [dispatch],
  });

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
        <Flex role="button" aria-label={t('controlLayers.fill.fillColor')} tabIndex={-1} minW={8} w={8} h={8}>
          <Tooltip label={tooltip}>
            <Flex alignItems="center" justifyContent="center" position="relative" w="full" h="full">
              <Box
                borderRadius="full"
                borderColor="base.600"
                w={6}
                h={6}
                borderWidth={2}
                bg={rgbaColorToString(bgColor)}
                position="absolute"
                top="0"
                left="0"
                zIndex={bgColorzIndex}
              />
              <Box
                borderRadius="full"
                borderColor="base.600"
                w={6}
                h={6}
                borderWidth={2}
                bg={rgbaColorToString(fgColor)}
                position="absolute"
                top="2"
                left="2"
                zIndex={fgColorzIndex}
              />
            </Flex>
          </Tooltip>
        </Flex>
      </PopoverTrigger>
      <Portal>
        <PopoverContent>
          <PopoverArrow />
          <PopoverBody minH={64}>
            <RgbaColorPicker color={activeColor} onChange={onColorChange} withNumberInput withSwatches />
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

ToolFillColorPicker.displayName = 'ToolFillColorPicker';
