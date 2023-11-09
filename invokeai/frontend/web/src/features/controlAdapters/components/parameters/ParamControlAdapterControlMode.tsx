import { useAppDispatch } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { useControlAdapterControlMode } from 'features/controlAdapters/hooks/useControlAdapterControlMode';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { controlAdapterControlModeChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { ControlMode } from 'features/controlAdapters/store/types';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export default function ParamControlAdapterControlMode({ id }: Props) {
  const isEnabled = useControlAdapterIsEnabled(id);
  const controlMode = useControlAdapterControlMode(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const CONTROL_MODE_DATA = [
    { label: t('controlnet.balanced'), value: 'balanced' },
    { label: t('controlnet.prompt'), value: 'more_prompt' },
    { label: t('controlnet.control'), value: 'more_control' },
    { label: t('controlnet.megaControl'), value: 'unbalanced' },
  ];

  const handleControlModeChange = useCallback(
    (controlMode: ControlMode) => {
      dispatch(controlAdapterControlModeChanged({ id, controlMode }));
    },
    [id, dispatch]
  );

  if (!controlMode) {
    return null;
  }

  return (
    <IAIInformationalPopover feature="controlNetControlMode">
      <IAIMantineSelect
        disabled={!isEnabled}
        label={t('controlnet.controlMode')}
        data={CONTROL_MODE_DATA}
        value={controlMode}
        onChange={handleControlModeChange}
      />
    </IAIInformationalPopover>
  );
}
