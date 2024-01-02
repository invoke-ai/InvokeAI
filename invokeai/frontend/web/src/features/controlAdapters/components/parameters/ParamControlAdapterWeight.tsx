import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
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

const formatValue = (v: number) => v.toFixed(2);

const ParamControlAdapterWeight = ({ id }: ParamControlAdapterWeightProps) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const weight = useControlAdapterWeight(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback(
    (weight: number) => {
      dispatch(controlAdapterWeightChanged({ id, weight }));
    },
    [dispatch, id]
  );
  const onReset = useCallback(() => {
    dispatch(controlAdapterWeightChanged({ id, weight: 1 }));
  }, [dispatch, id]);

  if (isNil(weight)) {
    // should never happen
    return null;
  }

  return (
    <InvControlGroup orientation="vertical">
      <InvControl
        label={t('controlnet.weight')}
        isDisabled={!isEnabled}
        feature="controlNetWeight"
      >
        <InvSlider
          value={weight}
          onChange={onChange}
          onReset={onReset}
          min={0}
          max={2}
          step={0.05}
          fineStep={0.01}
          marks={marks}
          formatValue={formatValue}
        />
        <InvNumberInput
          value={weight}
          onChange={onChange}
          onReset={onReset}
          min={-1}
          max={2}
          step={0.05}
          fineStep={0.01}
          maxW={20}
        />
      </InvControl>
    </InvControlGroup>
  );
};

export default memo(ParamControlAdapterWeight);

const marks = [0, 1, 2];
