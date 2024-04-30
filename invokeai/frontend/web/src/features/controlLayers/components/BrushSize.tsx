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
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { brushSizeChanged, initialRegionalPromptsState } from 'features/controlLayers/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [0, 100, 200, 300];
const formatPx = (v: number | string) => `${v} px`;

export const BrushSize = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const brushSize = useAppSelector((s) => s.regionalPrompts.present.brushSize);
  const onChange = useCallback(
    (v: number) => {
      dispatch(brushSizeChanged(Math.round(v)));
    },
    [dispatch]
  );
  return (
    <FormControl w="min-content">
      <FormLabel m={0}>{t('regionalPrompts.brushSize')}</FormLabel>
      <Popover isLazy>
        <PopoverTrigger>
          <CompositeNumberInput
            min={1}
            max={600}
            defaultValue={initialRegionalPromptsState.brushSize}
            value={brushSize}
            onChange={onChange}
            w={24}
            format={formatPx}
          />
        </PopoverTrigger>
        <PopoverContent w={200} py={2} px={4}>
          <PopoverArrow />
          <PopoverBody>
            <CompositeSlider
              min={1}
              max={300}
              defaultValue={initialRegionalPromptsState.brushSize}
              value={brushSize}
              onChange={onChange}
              marks={marks}
            />
          </PopoverBody>
        </PopoverContent>
      </Popover>
    </FormControl>
  );
});

BrushSize.displayName = 'BrushSize';
