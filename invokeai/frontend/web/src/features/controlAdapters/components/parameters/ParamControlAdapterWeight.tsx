import { useAppDispatch } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAISlider from 'common/components/IAISlider';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterWeight } from 'features/controlAdapters/hooks/useControlAdapterWeight';
import { controlAdapterWeightChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { isNil } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type ParamControlAdapterWeightProps = {
  id: string;
};

const ParamControlAdapterWeight = ({ id }: ParamControlAdapterWeightProps) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const weight = useControlAdapterWeight(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const handleWeightChanged = useCallback(
    (weight: number) => {
      dispatch(controlAdapterWeightChanged({ id, weight }));
    },
    [dispatch, id]
  );

  if (isNil(weight)) {
    // should never happen
    return null;
  }

  return (
    <IAIInformationalPopover feature="controlNetWeight">
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

export default memo(ParamControlAdapterWeight);
