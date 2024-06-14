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
import { brushWidthChanged } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [0, 100, 200, 300];
const formatPx = (v: number | string) => `${v} px`;

export const BrushWidth = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const width = useAppSelector((s) => s.canvasV2.tool.brush.width);
  const onChange = useCallback(
    (v: number) => {
      dispatch(brushWidthChanged(Math.round(v)));
    },
    [dispatch]
  );
  return (
    <FormControl w="min-content" gap={2}>
      <FormLabel m={0}>{t('controlLayers.brushWidth')}</FormLabel>
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

BrushWidth.displayName = 'BrushSize';
