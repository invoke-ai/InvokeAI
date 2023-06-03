import { useAppDispatch } from 'app/store/storeHooks';
import IAIFullCheckbox from 'common/components/IAIFullCheckbox';
import IAISwitch from 'common/components/IAISwitch';
import {
  controlNetToggled,
  isControlNetImagePreprocessedToggled,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';

type ParamControlNetIsEnabledProps = {
  controlNetId: string;
  isControlImageProcessed: boolean;
};

const ParamControlNetIsEnabled = (props: ParamControlNetIsEnabledProps) => {
  const { controlNetId, isControlImageProcessed } = props;
  const dispatch = useAppDispatch();

  const handleIsControlImageProcessedToggled = useCallback(() => {
    dispatch(
      isControlNetImagePreprocessedToggled({
        controlNetId,
      })
    );
  }, [controlNetId, dispatch]);

  return (
    <IAISwitch
      label="Preprocess"
      isChecked={isControlImageProcessed}
      onChange={handleIsControlImageProcessedToggled}
    />
  );
};

export default memo(ParamControlNetIsEnabled);
