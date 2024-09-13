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
          <IAIColorPicker color={fill} onChange={onChange} withNumberInput />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

ToolColorPicker.displayName = 'ToolFillColorPicker';
