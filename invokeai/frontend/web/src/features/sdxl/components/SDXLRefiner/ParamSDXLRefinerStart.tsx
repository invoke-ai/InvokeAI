import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setRefinerStart } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const selector = createMemoizedSelector([stateSelector], ({ sdxl }) => {
  const { refinerStart } = sdxl;
  return {
    refinerStart,
  };
});

const ParamSDXLRefinerStart = () => {
  const { refinerStart } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const isRefinerAvailable = useIsRefinerAvailable();
  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerStart(v)),
    [dispatch]
  );
  const { t } = useTranslation();

  const handleReset = useCallback(
    () => dispatch(setRefinerStart(0.8)),
    [dispatch]
  );

  return (
    <IAISlider
      label={t('sdxl.refinerStart')}
      step={0.01}
      min={0}
      max={1}
      onChange={handleChange}
      handleReset={handleReset}
      value={refinerStart}
      withInput
      withReset
      withSliderMarks
      isInteger={false}
      isDisabled={!isRefinerAvailable}
    />
  );
};

export default memo(ParamSDXLRefinerStart);
