import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { creativityChanged } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCreativity = () => {
  const creativity = useAppSelector((s) => s.upscale.creativity);
  const initial = 0;
  const sliderMin = -5;
  const sliderMax = 5;
  const numberInputMin = -5;
  const numberInputMax = 5;
  const coarseStep = 1;
  const fineStep = 1;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const marks = useMemo(() => [sliderMin, 0, sliderMax], [sliderMax, sliderMin]);
  const onChange = useCallback(
    (v: number) => {
      dispatch(creativityChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('upscaling.creativity')}</FormLabel>
      <CompositeSlider
        value={creativity}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        marks={marks}
      />
      <CompositeNumberInput
        value={creativity}
        defaultValue={initial}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamCreativity);
