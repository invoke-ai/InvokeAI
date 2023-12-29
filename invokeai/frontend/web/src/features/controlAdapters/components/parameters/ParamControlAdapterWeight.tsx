import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
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
    <InvControl
      label={t('controlnet.weight')}
      isDisabled={!isEnabled}
      feature="controlNetWeight"
      orientation="vertical"
    >
      <InvSlider
        value={weight}
        onChange={handleWeightChanged}
        min={0}
        max={2}
        step={0.01}
        marks={marks}
      />
    </InvControl>
  );
};

export default memo(ParamControlAdapterWeight);

const marks = [0, 1, 2];
