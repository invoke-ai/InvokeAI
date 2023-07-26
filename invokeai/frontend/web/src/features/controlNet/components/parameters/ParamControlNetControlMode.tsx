import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  ControlModes,
  controlNetControlModeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { useCallback, useMemo } from 'react';

type ParamControlNetControlModeProps = {
  controlNetId: string;
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
  const { controlNetId } = props;
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ controlNet }) => {
          const { controlMode, isEnabled } =
            controlNet.controlNets[controlNetId];
          return { controlMode, isEnabled };
        },
        defaultSelectorOptions
      ),
    [controlNetId]
  );

  const { controlMode, isEnabled } = useAppSelector(selector);

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
