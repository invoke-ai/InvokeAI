import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setCanvasCoherenceEdgeSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceEdgeSize = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceEdgeSize = useAppSelector((s) => s.generation.canvasCoherenceEdgeSize);
  const initial = useAppSelector((s) => s.config.sd.canvasCoherenceEdgeSize.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.canvasCoherenceEdgeSize.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.canvasCoherenceEdgeSize.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.canvasCoherenceEdgeSize.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.canvasCoherenceEdgeSize.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.canvasCoherenceEdgeSize.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.canvasCoherenceEdgeSize.fineStep);

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setCanvasCoherenceEdgeSize(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="compositingCoherenceEdgeSize">
        <FormLabel>{t('parameters.coherenceEdgeSize')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        value={canvasCoherenceEdgeSize}
        defaultValue={initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        value={canvasCoherenceEdgeSize}
        defaultValue={initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCanvasCoherenceEdgeSize);
