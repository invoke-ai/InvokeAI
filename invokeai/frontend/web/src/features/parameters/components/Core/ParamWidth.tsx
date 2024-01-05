import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(
  [stateSelector, selectOptimalDimension],
  ({ config }, optimalDimension) => {
    const { min, sliderMax, inputMax, fineStep, coarseStep } = config.sd.width;

    return {
      initial: optimalDimension,
      min,
      max: sliderMax,
      step: coarseStep,
      inputMax,
      fineStep,
    };
  }
);
export const ParamWidth = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const { initial, min, max, inputMax, step, fineStep } =
    useAppSelector(selector);

  const onChange = useCallback(
    (v: number) => {
      ctx.widthChanged(v);
    },
    [ctx]
  );

  const marks = useMemo(() => [min, initial, max], [min, initial, max]);

  return (
    <InvControl label={t('parameters.width')}>
      <InvSlider
        value={ctx.width}
        onChange={onChange}
        defaultValue={initial}
        min={min}
        max={max}
        step={step}
        fineStep={fineStep}
        marks={marks}
      />
      <InvNumberInput
        value={ctx.width}
        onChange={onChange}
        min={min}
        max={inputMax}
        step={step}
        fineStep={fineStep}
        defaultValue={initial}
      />
    </InvControl>
  );
});

ParamWidth.displayName = 'ParamWidth';
