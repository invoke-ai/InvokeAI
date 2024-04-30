import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
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
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useControlAdapterIsEnabled(id);
  const weight = useControlAdapterWeight(id);
  const initial = useAppSelector((s) => s.config.sd.ca.weight.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.ca.weight.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.ca.weight.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.ca.weight.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.ca.weight.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.ca.weight.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.ca.weight.fineStep);

  const onChange = useCallback(
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
    <FormControl isDisabled={!isEnabled} orientation="horizontal">
      <InformationalPopover feature="controlNetWeight">
        <FormLabel m={0}>{t('controlnet.weight')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={weight}
        onChange={onChange}
        defaultValue={initial}
        min={sliderMin}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        marks={marks}
        formatValue={formatValue}
      />
      <CompositeNumberInput
        value={weight}
        onChange={onChange}
        min={numberInputMin}
        max={numberInputMax}
        step={coarseStep}
        fineStep={fineStep}
        maxW={20}
        defaultValue={initial}
      />
    </FormControl>
  );
};

export default memo(ParamControlAdapterWeight);

const marks = [0, 1, 2];
