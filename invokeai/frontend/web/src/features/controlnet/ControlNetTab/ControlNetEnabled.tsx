import { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import { useAppDispatch } from 'app/store/storeHooks';
import IAICheckbox from 'common/components/IAICheckbox';
import { ChangeEvent } from 'react';
import { ControlNetConfig } from '../store/controlnetTypes';

interface ControlNetEnabledProps {
  controlnet: ControlNetConfig;
  setControlnet: ActionCreatorWithPayload<ControlNetConfig>;
}

export default function ControlNetEnabled(props: ControlNetEnabledProps) {
  const { controlnet, setControlnet } = props;

  const dispatch = useAppDispatch();

  const handleEnableChange = (e: ChangeEvent<HTMLInputElement>) => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetEnabled: e.currentTarget.checked,
      })
    );
  };

  return (
    <IAICheckbox
      label=""
      checked={controlnet.controlnetEnabled}
      onChange={handleEnableChange}
    />
  );
}
