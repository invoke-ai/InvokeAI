import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISlider from 'common/components/IAISlider';
import { controlNetWeightChanged } from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback, useMemo } from 'react';

type ParamControlNetWeightProps = {
  controlNetId: string;
};

const ParamControlNetWeight = (props: ParamControlNetWeightProps) => {
  const { controlNetId } = props;
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlNet }) => controlNet.controlNets[controlNetId]?.weight,
        defaultSelectorOptions
      ),
    [controlNetId]
  );

  const weight = useAppSelector(selector);
  const handleWeightChanged = useCallback(
    (weight: number) => {
      dispatch(controlNetWeightChanged({ controlNetId, weight }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAISlider
      label={'Weight'}
      value={weight}
      onChange={handleWeightChanged}
      min={0}
      max={2}
      step={0.01}
      withSliderMarks
      sliderMarks={[0, 1, 2]}
    />
  );
};

export default memo(ParamControlNetWeight);
