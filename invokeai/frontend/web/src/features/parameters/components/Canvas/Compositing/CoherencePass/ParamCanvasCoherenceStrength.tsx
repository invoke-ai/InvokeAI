import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setCanvasCoherenceStrength } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceStrength = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceStrength = useAppSelector((s) => s.generation.canvasCoherenceStrength);
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setCanvasCoherenceStrength(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="compositingStrength">
        <FormLabel>{t('parameters.coherenceStrength')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={0}
        max={1}
        step={0.01}
        value={canvasCoherenceStrength}
        defaultValue={0.75}
        onChange={handleChange}
      />
      <CompositeNumberInput
        min={0}
        max={1}
        step={0.01}
        value={canvasCoherenceStrength}
        defaultValue={0.75}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCanvasCoherenceStrength);
