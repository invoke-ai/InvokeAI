import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setCanvasCoherenceSteps } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceSteps = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceSteps = useAppSelector((s) => s.generation.canvasCoherenceSteps);
  const initial = useAppSelector((s) => s.config.sd.canvasCoherenceSteps.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.canvasCoherenceSteps.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.canvasCoherenceSteps.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.canvasCoherenceSteps.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.canvasCoherenceSteps.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.canvasCoherenceSteps.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.canvasCoherenceSteps.fineStep);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setCanvasCoherenceSteps(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="compositingCoherenceSteps">
        <FormLabel>{t('parameters.coherenceSteps')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={canvasCoherenceSteps}
        defaultValue={initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        value={canvasCoherenceSteps}
        defaultValue={initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCanvasCoherenceSteps);
