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
import { selectCanvasSettingsSlice, settingsColorChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectColor = createSelector(selectCanvasSettingsSlice, (settings) => settings.color);

export const ToolColorPicker = memo(() => {
  const { t } = useTranslation();
  const fill = useAppSelector(selectColor);
  const dispatch = useAppDispatch();
  const onChange = useCallback(
    (color: RgbaColor) => {
      dispatch(settingsColorChanged(color));
    },
    [dispatch]
  );
  const imageViewer = useImageViewer();

  useRegisteredHotkeys({
    id: 'setFillToWhite',
    category: 'canvas',
    callback: () => dispatch(settingsColorChanged({ r: 255, g: 255, b: 255, a: 1 })),
    options: { preventDefault: true, enabled: !imageViewer.isOpen },
    dependencies: [dispatch, imageViewer.isOpen],
  });

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Flex role="button" aria-label={t('controlLayers.fill.fillColor')} tabIndex={-1} w={8} h={8}>
          <Tooltip label={t('controlLayers.fill.fillColor')}>
            <Flex w="full" h="full" alignItems="center" justifyContent="center">
              <Box
                borderRadius="full"
                borderColor="base.600"
                w={6}
                h={6}
                borderWidth={2}
                bg={rgbaColorToString(fill)}
              />
            </Flex>
          </Tooltip>
        </Flex>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody minH={64}>
          <RgbaColorPicker color={fill} onChange={onChange} withNumberInput withSwatches />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

ToolColorPicker.displayName = 'ToolFillColorPicker';
