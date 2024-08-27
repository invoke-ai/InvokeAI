import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { scaleChanged, selectUpscaleScale } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [2, 4, 8];

const formatValue = (val: string | number) => `${val}x`;

export const UpscaleScaleSlider = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const scale = useAppSelector(selectUpscaleScale);

  const onChange = useCallback(
    (val: number) => {
      dispatch(scaleChanged(val));
    },
    [dispatch]
  );

  return (
    <FormControl orientation="vertical" gap={0}>
      <InformationalPopover feature="scale">
        <FormLabel m={0}>{t('upscaling.scale')}</FormLabel>
      </InformationalPopover>
      <Flex w="full" gap={4}>
        <CompositeSlider
          min={2}
          max={8}
          value={scale}
          onChange={onChange}
          marks={marks}
          formatValue={formatValue}
          defaultValue={4}
        />
        <CompositeNumberInput
          maxW={20}
          value={scale}
          onChange={onChange}
          defaultValue={4}
          min={2}
          max={16}
          format={formatValue}
        />
      </Flex>
    </FormControl>
  );
});

UpscaleScaleSlider.displayName = 'UpscaleScaleSlider';
