import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import {
  ControlNetConfig,
  controlNetAutoConfigToggled,
} from 'features/controlNet/store/controlNetSlice';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { memo, useCallback } from 'react';

type Props = {
  controlNet: ControlNetConfig;
};

const ParamControlNetShouldAutoConfig = (props: Props) => {
  const { controlNetId, isEnabled, shouldAutoConfig } = props.controlNet;
  const dispatch = useAppDispatch();
  const isBusy = useAppSelector(selectIsBusy);

  const handleShouldAutoConfigChanged = useCallback(() => {
    dispatch(controlNetAutoConfigToggled({ controlNetId }));
  }, [controlNetId, dispatch]);

  return (
    <IAISwitch
      label="Auto configure processor"
      aria-label="Auto configure processor"
      isChecked={shouldAutoConfig}
      onChange={handleShouldAutoConfigChanged}
      isDisabled={isBusy || !isEnabled}
    />
  );
};

export default memo(ParamControlNetShouldAutoConfig);
