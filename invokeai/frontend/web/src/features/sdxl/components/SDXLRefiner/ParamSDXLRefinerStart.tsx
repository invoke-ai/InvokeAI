import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setRefinerStart } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector([stateSelector], ({ sdxl }) => {
  const { refinerStart } = sdxl;
  return {
    refinerStart,
  };
});

const ParamSDXLRefinerStart = () => {
  const { refinerStart } = useAppSelector(selector);
  const dispatch = useAppDispatch();
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
    <InvControl label={t('sdxl.refinerStart')}>
      <InvSlider
        step={0.01}
        min={0}
        max={1}
        onChange={handleChange}
        onReset={handleReset}
        value={refinerStart}
        withNumberInput
        marks
      />
    </InvControl>
  );
};

export default memo(ParamSDXLRefinerStart);
