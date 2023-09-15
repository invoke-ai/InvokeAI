import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  ControlModes,
  ControlNetConfig,
  controlNetControlModeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { ControlNetControlModePopover } from 'features/informationalPopovers/components/controlNetControlMode';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type ParamControlNetControlModeProps = {
  controlNet: ControlNetConfig;
};

export default function ParamControlNetControlMode(
  props: ParamControlNetControlModeProps
) {
  const { controlMode, isEnabled, controlNetId } = props.controlNet;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const CONTROL_MODE_DATA = [
    { label: t('controlnet.balanced'), value: 'balanced' },
    { label: t('controlnet.prompt'), value: 'more_prompt' },
    { label: t('controlnet.control'), value: 'more_control' },
    { label: t('controlnet.megaControl'), value: 'unbalanced' },
  ];

  const handleControlModeChange = useCallback(
    (controlMode: ControlModes) => {
      dispatch(controlNetControlModeChanged({ controlNetId, controlMode }));
    },
    [controlNetId, dispatch]
  );

  return (
    <ControlNetControlModePopover>
      <IAIMantineSelect
        disabled={!isEnabled}
        label={t('controlnet.controlMode')}
        data={CONTROL_MODE_DATA}
        value={String(controlMode)}
        onChange={handleControlModeChange}
      />
    </ControlNetControlModePopover>
  );
}
