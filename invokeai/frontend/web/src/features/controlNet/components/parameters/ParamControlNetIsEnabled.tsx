import { useAppDispatch } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { controlNetToggled } from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type ParamControlNetIsEnabledProps = {
  controlNetId: string;
  isEnabled: boolean;
};

const ParamControlNetIsEnabled = (props: ParamControlNetIsEnabledProps) => {
  const { controlNetId, isEnabled } = props;
  const dispatch = useAppDispatch();

  const handleIsEnabledChanged = useCallback(() => {
    dispatch(controlNetToggled({ controlNetId }));
  }, [dispatch, controlNetId]);

  return (
    <IAISwitch
      label="Enabled"
      isChecked={isEnabled}
      onChange={handleIsEnabledChanged}
    />
  );
};

export default memo(ParamControlNetIsEnabled);
