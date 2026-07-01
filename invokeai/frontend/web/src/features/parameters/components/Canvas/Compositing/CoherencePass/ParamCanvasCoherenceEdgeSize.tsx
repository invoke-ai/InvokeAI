import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCanvasCoherenceEdgeSize, setCanvasCoherenceEdgeSize } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 16,
  sliderMin: 0,
  sliderMax: 128,
  numberInputMin: 0,
  numberInputMax: 1024,
  fineStep: 8,
  coarseStep: 16,
};

const ParamCanvasCoherenceEdgeSize = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceEdgeSize = useAppSelector(selectCanvasCoherenceEdgeSize);

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
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        value={canvasCoherenceEdgeSize}
        defaultValue={CONSTRAINTS.initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        value={canvasCoherenceEdgeSize}
        defaultValue={CONSTRAINTS.initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCanvasCoherenceEdgeSize);
