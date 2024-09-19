import { Box, Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger, Tooltip } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { selectCanvasSettingsSlice, settingsColorChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectColor = createSelector(selectCanvasSettingsSlice, (settings) => settings.color);

const SWATCHES = [
  { r: 0, g: 0, b: 0, a: 1 }, // black
  { r: 255, g: 255, b: 255, a: 1 }, // white
  { r: 255, g: 90, b: 94, a: 1 }, // red
  { r: 255, g: 146, b: 75, a: 1 }, // orange
  { r: 255, g: 202, b: 59, a: 1 }, // yellow
  { r: 197, g: 202, b: 48, a: 1 }, // lime
  { r: 138, g: 201, b: 38, a: 1 }, // green
  { r: 83, g: 165, b: 117, a: 1 }, // teal
  { r: 23, g: 130, b: 196, a: 1 }, // blue
  { r: 66, g: 103, b: 172, a: 1 }, // indigo
  { r: 107, g: 76, b: 147, a: 1 }, // purple
];

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
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Flex role="button" aria-label={t('controlLayers.fill.fillColor')} tabIndex={-1} w={8} h={8}>
          <Tooltip label={t('controlLayers.fill.fillColor')}>
            <Flex w="full" h="full" alignItems="center" justifyContent="center">
              <Box
                borderRadius="full"
                borderColor="base.300"
                w={6}
                h={6}
                borderWidth={1}
                bg={rgbaColorToString(fill)}
              />
            </Flex>
          </Tooltip>
        </Flex>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <Flex flexDir="column" gap={4}>
          <IAIColorPicker color={fill} onChange={onChange} withNumberInput />
            <Flex gap={2} justifyContent="space-between">
              {SWATCHES.map((color, i) => (
                <ColorSwatch key={i} color={color} />
              ))}
            </Flex>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

ToolColorPicker.displayName = 'ToolFillColorPicker';

const ColorSwatch = ({ color }: { color: RgbaColor }) => {
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(settingsColorChanged(color));
  }, [color, dispatch]);
  return (
    <Box
      role="button"
      onClick={onClick}
      h={8}
      w={8}
      borderColor="base.300"
      borderWidth={1}
      bg={rgbaColorToString(color)}
      borderRadius="base"
    />
  );
};
