import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOnChange } from 'common/components/InvSelect/types';
import { useControlAdapterControlMode } from 'features/controlAdapters/hooks/useControlAdapterControlMode';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { controlAdapterControlModeChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlMode } from 'features/controlAdapters/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

const ParamControlAdapterControlMode = ({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const controlMode = useControlAdapterControlMode(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const CONTROL_MODE_DATA = useMemo(
    () => [
      { label: t('controlnet.balanced'), value: 'balanced' },
      { label: t('controlnet.prompt'), value: 'more_prompt' },
      { label: t('controlnet.control'), value: 'more_control' },
      { label: t('controlnet.megaControl'), value: 'unbalanced' },
    ],
    [t]
  );

  const handleControlModeChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      dispatch(
        controlAdapterControlModeChanged({
          id,
          controlMode: v.value as ControlMode,
        })
      );
    },
    [id, dispatch]
  );

  const value = useMemo(
    () => CONTROL_MODE_DATA.filter((o) => o.value === controlMode)[0],
    [CONTROL_MODE_DATA, controlMode]
  );

  if (!controlMode) {
    return null;
  }

  return (
    <InvControl
      isDisabled={!isEnabled}
      label={t('controlnet.controlMode')}
      feature="controlNetControlMode"
    >
      <InvSelect
        value={value}
        options={CONTROL_MODE_DATA}
        onChange={handleControlModeChange}
      />
    </InvControl>
  );
};

export default memo(ParamControlAdapterControlMode);
