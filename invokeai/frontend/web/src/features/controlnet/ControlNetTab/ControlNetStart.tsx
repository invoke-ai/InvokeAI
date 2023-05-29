import { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import { useAppDispatch } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { ControlNetConfig } from '../store/controlnetTypes';

interface ControlNetStartProps {
  controlnet: ControlNetConfig;
  setControlnet: ActionCreatorWithPayload<ControlNetConfig>;
}

export default function ControlNetStart(props: ControlNetStartProps) {
  const { controlnet, setControlnet } = props;

  const dispatch = useAppDispatch();

  const handleStartChange = (v: number) => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetStart: v,
      })
    );
  };

  const handleStartReset = () => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetStart: 0,
      })
    );
  };

  return (
    <IAISlider
      label="Start"
      value={controlnet.controlnetStart}
      onChange={handleStartChange}
      withInput
      withReset
      handleReset={handleStartReset}
      withSliderMarks
      min={0}
      max={1}
      step={0.1}
    />
  );
}
