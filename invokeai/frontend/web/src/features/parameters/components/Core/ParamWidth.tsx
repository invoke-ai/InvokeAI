import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamWidth = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const min = useAppSelector((s) => s.config.sd.width.min);
  const sliderMax = useAppSelector((s) => s.config.sd.width.sliderMax);
  const inputMax = useAppSelector((s) => s.config.sd.width.inputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.width.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.width.fineStep);

  const onChange = useCallback(
    (v: number) => {
      ctx.widthChanged(v);
    },
    [ctx]
  );

  const marks = useMemo(
    () => [min, optimalDimension, sliderMax],
    [min, optimalDimension, sliderMax]
  );

  return (
    <InvControl label={t('parameters.width')}>
      <InvSlider
        value={ctx.width}
        onChange={onChange}
        defaultValue={optimalDimension}
        min={min}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        marks={marks}
      />
      <InvNumberInput
        value={ctx.width}
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

ParamWidth.displayName = 'ParamWidth';
