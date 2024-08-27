import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCanvasCoherenceEdgeSize, setCanvasCoherenceEdgeSize } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasCoherenceEdgeSizeConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceEdgeSize = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceEdgeSize = useAppSelector(selectCanvasCoherenceEdgeSize);
  const config = useAppSelector(selectCanvasCoherenceEdgeSizeConfig);

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
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        value={canvasCoherenceEdgeSize}
        defaultValue={config.initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        value={canvasCoherenceEdgeSize}
        defaultValue={config.initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCanvasCoherenceEdgeSize);
