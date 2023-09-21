import { useAppDispatch } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import {
  ControlNetConfig,
  controlNetWeightChanged,
} from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type ParamControlNetWeightProps = {
  controlNet: ControlNetConfig;
};

const ParamControlNetWeight = (props: ParamControlNetWeightProps) => {
  const { weight, isEnabled, controlNetId } = props.controlNet;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const handleWeightChanged = useCallback(
    (weight: number) => {
      dispatch(controlNetWeightChanged({ controlNetId, weight }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAIInformationalPopover details="controlNetWeight">
      <IAISlider
        isDisabled={!isEnabled}
        label={t('controlnet.weight')}
        value={weight}
        onChange={handleWeightChanged}
        min={0}
        max={2}
        step={0.01}
        withSliderMarks
        sliderMarks={[0, 1, 2]}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamControlNetWeight);
