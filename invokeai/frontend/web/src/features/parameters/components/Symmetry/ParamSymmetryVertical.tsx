import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setVerticalSymmetrySteps } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSymmetryVertical = () => {
  const verticalSymmetrySteps = useAppSelector(
    (state: RootState) => state.generation.verticalSymmetrySteps
  );

  const steps = useAppSelector((state: RootState) => state.generation.steps);

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setVerticalSymmetrySteps(v));
    },
    [dispatch]
  );
  const handleReset = useCallback(() => {
    dispatch(setVerticalSymmetrySteps(0));
  }, [dispatch]);

  return (
    <InvControl label={t('parameters.vSymmetryStep')}>
      <InvSlider
        value={verticalSymmetrySteps}
        onChange={handleChange}
        min={0}
        max={steps}
        step={1}
        withNumberInput
        marks
        onReset={handleReset}
      />
    </InvControl>
  );
};

export default memo(ParamSymmetryVertical);
