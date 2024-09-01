import {
  CompositeNumberInput,
  CompositeSlider,
  FormControl,
  FormLabel,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { brushWidthChanged, selectToolSlice } from 'features/controlLayers/store/toolSlice';
import { clamp } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

const marks = [1, 100, 200, 300];
const formatPx = (v: number | string) => `${v} px`;
const selectBrushWidth = createSelector(selectToolSlice, (tool) => tool.brush.width);

export const ToolBrushWidth = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('brush');
  const width = useAppSelector(selectBrushWidth);
  const onChange = useCallback(
    (v: number) => {
      dispatch(brushWidthChanged(clamp(Math.round(v), 1, 600)));
    },
    [dispatch]
  );

  const increment = useCallback(() => {
    let newWidth = Math.round(width * 1.15);
    if (newWidth === width) {
      newWidth += 1;
    }
    onChange(newWidth);
  }, [onChange, width]);

  const decrement = useCallback(() => {
    let newWidth = Math.round(width * 0.85);
    if (newWidth === width) {
      newWidth -= 1;
    }
    onChange(newWidth);
  }, [onChange, width]);

  useHotkeys('[', decrement, { enabled: isSelected }, [decrement, isSelected]);
  useHotkeys(']', increment, { enabled: isSelected }, [increment, isSelected]);

  return (
    <FormControl w="min-content" gap={2}>
      <FormLabel m={0}>{t('controlLayers.width')}</FormLabel>
      <Popover isLazy>
        <PopoverTrigger>
          <CompositeNumberInput
            min={1}
            max={600}
            defaultValue={50}
            value={width}
            onChange={onChange}
            w={24}
            format={formatPx}
          />
        </PopoverTrigger>
        <PopoverContent w={200} py={2} px={4}>
          <PopoverArrow />
          <PopoverBody>
            <CompositeSlider min={1} max={300} defaultValue={50} value={width} onChange={onChange} marks={marks} />
          </PopoverBody>
        </PopoverContent>
      </Popover>
    </FormControl>
  );
});

ToolBrushWidth.displayName = 'ToolBrushWidth';
