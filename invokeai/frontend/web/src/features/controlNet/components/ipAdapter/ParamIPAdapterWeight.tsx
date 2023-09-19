import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { ipAdapterWeightChanged } from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamIPAdapterWeight = () => {
  const isIpAdapterEnabled = useAppSelector(
    (state: RootState) => state.controlNet.isIPAdapterEnabled
  );
  const ipAdapterWeight = useAppSelector(
    (state: RootState) => state.controlNet.ipAdapterInfo.weight
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleWeightChanged = useCallback(
    (weight: number) => {
      dispatch(ipAdapterWeightChanged(weight));
    },
    [dispatch]
  );

  const handleWeightReset = useCallback(() => {
    dispatch(ipAdapterWeightChanged(1));
  }, [dispatch]);

  return (
    <IAISlider
      isDisabled={!isIpAdapterEnabled}
      label={t('controlnet.weight')}
      value={ipAdapterWeight}
      onChange={handleWeightChanged}
      min={0}
      max={2}
      step={0.01}
      withSliderMarks
      sliderMarks={[0, 1, 2]}
      withReset
      handleReset={handleWeightReset}
    />
  );
};

export default memo(ParamIPAdapterWeight);
