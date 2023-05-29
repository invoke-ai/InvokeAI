import { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import { useAppDispatch } from 'app/store/storeHooks';
import IAICustomSelect from 'common/components/IAICustomSelect';
import {
  ControlNetConfig,
  ControlNetProcessorTypes,
} from '../store/controlnetTypes';
import { CONTROLNET_PROCESSORS } from '../utils/controlnetConstants';

interface ControlNetProcessorProps {
  controlnet: ControlNetConfig;
  setControlnet: ActionCreatorWithPayload<ControlNetConfig>;
}

export default function ControlNetProcessor(props: ControlNetProcessorProps) {
  const { controlnet, setControlnet } = props;
  const dispatch = useAppDispatch();

  const handleProcessorChange = (v: string | null | void) => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetProcessor: v as ControlNetProcessorTypes,
      })
    );
  };

  return (
    <IAICustomSelect
      label="Processor"
      selectedItem={controlnet.controlnetProcessor}
      items={CONTROLNET_PROCESSORS}
      setSelectedItem={handleProcessorChange}
    />
  );
}
