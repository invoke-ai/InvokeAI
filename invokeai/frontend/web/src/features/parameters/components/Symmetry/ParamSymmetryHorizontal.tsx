import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setHorizontalSymmetrySteps } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSymmetryHorizontal = () => {
  const horizontalSymmetrySteps = useAppSelector(
    (s) => s.generation.horizontalSymmetrySteps
  );

  const steps = useAppSelector((s) => s.generation.steps);

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setHorizontalSymmetrySteps(v));
    },
    [dispatch]
  );

  return (
    <InvControl label={t('parameters.hSymmetryStep')}>
      <InvSlider
        value={horizontalSymmetrySteps}
        defaultValue={0}
        onChange={handleChange}
        min={0}
        max={steps}
        step={1}
        withNumberInput
        marks
      />
    </InvControl>
  );
};

export default memo(ParamSymmetryHorizontal);
