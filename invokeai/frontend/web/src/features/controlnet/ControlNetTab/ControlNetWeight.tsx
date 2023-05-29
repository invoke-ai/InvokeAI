import { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import { useAppDispatch } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { ControlNetConfig } from '../store/controlnetTypes';

interface ControlNetWeightrProps {
  controlnet: ControlNetConfig;
  setControlnet: ActionCreatorWithPayload<ControlNetConfig>;
}

export default function ControlNetWeight(props: ControlNetWeightrProps) {
  const { controlnet, setControlnet } = props;

  const dispatch = useAppDispatch();

  const handleWeightChange = (v: number) => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetWeight: v,
      })
    );
  };

  const handleWeightReset = () => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetWeight: 1,
      })
    );
  };

  return (
    <IAISlider
      label="Weight"
      value={controlnet.controlnetWeight}
      onChange={handleWeightChange}
      withInput
      withReset
      handleReset={handleWeightReset}
      withSliderMarks
      min={0}
      max={1}
      step={0.1}
    />
  );
}
