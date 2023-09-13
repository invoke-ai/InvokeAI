import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import {
  ControlNetConfig,
  controlNetAutoConfigToggled,
} from 'features/controlNet/store/controlNetSlice';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  controlNet: ControlNetConfig;
};

const ParamControlNetShouldAutoConfig = (props: Props) => {
  const { controlNetId, isEnabled, shouldAutoConfig } = props.controlNet;
  const dispatch = useAppDispatch();
  const isBusy = useAppSelector(selectIsBusy);
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
      isDisabled={isBusy || !isEnabled}
    />
  );
};

export default memo(ParamControlNetShouldAutoConfig);
