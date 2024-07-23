import { CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { scaleChanged } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [2, 4, 8, 16];

const formatValue = (val: number) => `${val}x`;

export const UpscaleScaleSlider = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const scale = useAppSelector((s) => s.upscale.scale);

  const onChange = useCallback(
    (val: number) => {
      dispatch(scaleChanged(val));
    },
    [dispatch]
  );

  return (
    <FormControl orientation="vertical" gap={0}>
      <FormLabel m={0}>{t('upscaling.scale')}</FormLabel>
      <CompositeSlider
        min={2}
        max={16}
        value={scale}
        onChange={onChange}
        marks={marks}
        formatValue={formatValue}
        withThumbTooltip
      />
    </FormControl>
  );
});

UpscaleScaleSlider.displayName = 'UpscaleScaleSlider';
