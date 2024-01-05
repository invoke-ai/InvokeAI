import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamHeight = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const min = useAppSelector((s) => s.config.sd.height.min);
  const sliderMax = useAppSelector((s) => s.config.sd.height.sliderMax);
  const inputMax = useAppSelector((s) => s.config.sd.height.inputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.height.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.height.fineStep);

  const onChange = useCallback(
    (v: number) => {
      ctx.heightChanged(v);
    },
    [ctx]
  );

  const marks = useMemo(
    () => [min, optimalDimension, sliderMax],
    [min, optimalDimension, sliderMax]
  );

  return (
    <InvControl label={t('parameters.height')}>
      <InvSlider
        value={ctx.height}
        defaultValue={optimalDimension}
        onChange={onChange}
        min={min}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        marks={marks}
      />
      <InvNumberInput
        value={ctx.height}
        onChange={onChange}
        min={min}
        max={inputMax}
        step={coarseStep}
        fineStep={fineStep}
        defaultValue={optimalDimension}
      />
    </InvControl>
  );
});

ParamHeight.displayName = 'ParamHeight';
