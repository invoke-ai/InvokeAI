import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setCanvasCoherenceMinDenoise } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceMinDenoise = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceMinDenoise = useAppSelector((s) => s.generation.canvasCoherenceMinDenoise);
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setCanvasCoherenceMinDenoise(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="compositingCoherenceMinDenoise">
        <FormLabel>{t('parameters.coherenceMinDenoise')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={0}
        max={1}
        step={0.01}
        value={canvasCoherenceMinDenoise}
        defaultValue={0}
        onChange={handleChange}
      />
      <CompositeNumberInput
        min={0}
        max={1}
        step={0.01}
        value={canvasCoherenceMinDenoise}
        defaultValue={0}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamCanvasCoherenceMinDenoise);
