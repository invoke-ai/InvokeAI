import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  ControlModes,
  ControlNetConfig,
  controlNetControlModeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { useCallback } from 'react';

type ParamControlNetControlModeProps = {
  controlNet: ControlNetConfig;
};

const CONTROL_MODE_DATA = [
  { label: 'Balanced', value: 'balanced' },
  { label: 'Prompt', value: 'more_prompt' },
  { label: 'Control', value: 'more_control' },
  { label: 'Mega Control', value: 'unbalanced' },
];

export default function ParamControlNetControlMode(
  props: ParamControlNetControlModeProps
) {
  const { controlMode, isEnabled, controlNetId } = props.controlNet;
  const dispatch = useAppDispatch();

  const handleControlModeChange = useCallback(
    (controlMode: ControlModes) => {
      dispatch(controlNetControlModeChanged({ controlNetId, controlMode }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAIMantineSelect
      disabled={!isEnabled}
      label="Control Mode"
      data={CONTROL_MODE_DATA}
      value={String(controlMode)}
      onChange={handleControlModeChange}
    />
  );
}
