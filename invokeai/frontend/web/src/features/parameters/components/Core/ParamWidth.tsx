import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamWidth = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const sliderMin = useAppSelector((s) => s.config.sd.width.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.width.sliderMax);
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.width.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.width.numberInputMax
  );
  const coarseStep = useAppSelector((s) => s.config.sd.width.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.width.fineStep);

  const onChange = useCallback(
    (v: number) => {
      ctx.widthChanged(v);
    },
    [ctx]
  );

  const marks = useMemo(
    () => [sliderMin, optimalDimension, sliderMax],
    [sliderMin, optimalDimension, sliderMax]
  );

  return (
    <InvControl label={t('parameters.width')}>
      <InvSlider
        value={ctx.width}
        onChange={onChange}
        defaultValue={optimalDimension}
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

ParamWidth.displayName = 'ParamWidth';
