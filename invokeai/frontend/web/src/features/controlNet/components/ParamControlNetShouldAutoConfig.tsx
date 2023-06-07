import { useAppDispatch } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { controlNetAutoConfigToggled } from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type Props = {
  controlNetId: string;
  shouldAutoConfig: boolean;
};

const ParamControlNetShouldAutoConfig = (props: Props) => {
  const { controlNetId, shouldAutoConfig } = props;
  const dispatch = useAppDispatch();

  const handleShouldAutoConfigChanged = useCallback(() => {
    dispatch(controlNetAutoConfigToggled({ controlNetId }));
  }, [controlNetId, dispatch]);

  return (
    <IAISwitch
      label="Auto configure processor"
      aria-label="Auto configure processor"
      isChecked={shouldAutoConfig}
      onChange={handleShouldAutoConfigChanged}
    />
  );
};

export default memo(ParamControlNetShouldAutoConfig);
