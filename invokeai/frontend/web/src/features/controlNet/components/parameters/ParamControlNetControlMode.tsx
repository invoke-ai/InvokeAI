import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  ControlModes,
  controlNetControlModeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type ParamControlNetControlModeProps = {
  controlNetId: string;
  controlMode: string;
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
  const { controlNetId, controlMode = false } = props;
  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const handleControlModeChange = useCallback(
    (controlMode: ControlModes) => {
      dispatch(controlNetControlModeChanged({ controlNetId, controlMode }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAIMantineSelect
      label={t('parameters.controlNetControlMode')}
      data={CONTROL_MODE_DATA}
      value={String(controlMode)}
      onChange={handleControlModeChange}
    />
  );
}
