import { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import { useAppDispatch } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { ControlNetConfig } from '../store/controlnetTypes';

interface ControlNetEndProps {
  controlnet: ControlNetConfig;
  setControlnet: ActionCreatorWithPayload<ControlNetConfig>;
}

export default function ControlNetEnd(props: ControlNetEndProps) {
  const { controlnet, setControlnet } = props;

  const dispatch = useAppDispatch();

  const handleEndChange = (v: number) => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetEnd: v,
      })
    );
  };

  const handleEndReset = () => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetEnd: 1,
      })
    );
  };

  return (
    <IAISlider
      label="End"
      value={controlnet.controlnetEnd}
      onChange={handleEndChange}
      withInput
      withReset
      handleReset={handleEndReset}
      withSliderMarks
      min={0}
      max={1}
      step={0.1}
    />
  );
}
