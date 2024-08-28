import { Box, Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger, Tooltip } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { fillChanged, selectToolSlice } from 'features/controlLayers/store/toolSlice';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectFill = createSelector(selectToolSlice, (tool) => tool.fill);

export const ToolFillColorPicker = memo(() => {
  const { t } = useTranslation();
  const fill = useAppSelector(selectFill);
  const dispatch = useAppDispatch();
  const onChange = useCallback(
    (color: RgbaColor) => {
      dispatch(fillChanged(color));
    },
    [dispatch]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Flex role="button" aria-label={t('controlLayers.fill.fillColor')} tabIndex={-1} w={8} h={8}>
          <Tooltip label={t('controlLayers.fill.fillColor')}>
            <Flex w="full" h="full" alignItems="center" justifyContent="center">
              <Box borderRadius="full" w={6} h={6} borderWidth={1} bg={rgbaColorToString(fill)} />
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

ToolFillColorPicker.displayName = 'ToolFillColorPicker';
