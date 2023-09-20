import { useAppDispatch } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import {
  ControlNetConfig,
  controlNetAutoConfigToggled,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  controlNet: ControlNetConfig;
};

const ParamControlNetShouldAutoConfig = (props: Props) => {
  const { controlNetId, isEnabled, shouldAutoConfig } = props.controlNet;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleShouldAutoConfigChanged = useCallback(() => {
    dispatch(controlNetAutoConfigToggled({ controlNetId }));
  }, [controlNetId, dispatch]);

  return (
    <IAISwitch
      label={t('controlnet.autoConfigure')}
      aria-label={t('controlnet.autoConfigure')}
      isChecked={shouldAutoConfig}
      onChange={handleShouldAutoConfigChanged}
      isDisabled={!isEnabled}
    />
  );
};

export default memo(ParamControlNetShouldAutoConfig);
