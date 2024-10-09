import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  weight: number;
  onChange: (weight: number) => void;
};

const formatValue = (v: number) => v.toFixed(2);
const marks = [0, 1, 2];

const selectWeightConfig = createSelector(selectConfigSlice, (config) => config.sd.ca.weight);

export const Weight = memo(({ weight, onChange }: Props) => {
  const { t } = useTranslation();
  const config = useAppSelector(selectWeightConfig);

  return (
    <FormControl orientation="horizontal">
      <InformationalPopover feature="controlNetWeight">
        <FormLabel m={0}>{t('controlLayers.weight')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={weight}
        onChange={onChange}
        defaultValue={config.initial}
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        marks={marks}
        formatValue={formatValue}
      />
      <CompositeNumberInput
        value={weight}
        onChange={onChange}
        min={config.numberInputMin}
        max={config.numberInputMax}
        step={config.coarseStep}
        fineStep={config.fineStep}
        maxW={20}
        defaultValue={config.initial}
      />
    </FormControl>
  );
});

Weight.displayName = 'Weight';
