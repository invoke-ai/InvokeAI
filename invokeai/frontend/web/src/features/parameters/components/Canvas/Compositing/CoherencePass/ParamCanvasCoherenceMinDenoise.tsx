import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectCanvasCoherenceMinDenoise,
  setCanvasCoherenceMinDenoise,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceMinDenoise = () => {
  const dispatchParams = useParamsDispatch();
  const canvasCoherenceMinDenoise = useAppSelector(selectCanvasCoherenceMinDenoise);
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatchParams(setCanvasCoherenceMinDenoise, v);
    },
    [dispatchParams]
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
