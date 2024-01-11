import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamHeight = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const sliderMin = useAppSelector((s) => s.config.sd.height.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.height.sliderMax);
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.height.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.height.numberInputMax
  );
  const coarseStep = useAppSelector((s) => s.config.sd.height.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.height.fineStep);

  const onChange = useCallback(
    (v: number) => {
      ctx.heightChanged(v);
    },
    [ctx]
  );

  const marks = useMemo(
    () => [sliderMin, optimalDimension, sliderMax],
    [sliderMin, optimalDimension, sliderMax]
  );

  return (
    <InvControl label={t('parameters.height')}>
      <InvSlider
        value={ctx.height}
        defaultValue={optimalDimension}
        onChange={onChange}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        marks={marks}
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
      />
    </InvControl>
  );
});

ParamHeight.displayName = 'ParamHeight';
