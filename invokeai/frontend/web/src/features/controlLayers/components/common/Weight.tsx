import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 1,
  sliderMin: 0,
  sliderMax: 2,
  numberInputMin: -1,
  numberInputMax: 2,
  fineStep: 0.01,
  coarseStep: 0.05,
};

type Props = {
  weight: number;
  onChange: (weight: number) => void;
};

const formatValue = (v: number) => v.toFixed(2);
const marks = [0, 1, 2];

export const Weight = memo(({ weight, onChange }: Props) => {
  const { t } = useTranslation();

  return (
    <FormControl orientation="horizontal">
      <InformationalPopover feature="controlNetWeight">
        <FormLabel m={0}>{t('controlLayers.weight')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={weight}
        onChange={onChange}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        marks={marks}
        formatValue={formatValue}
      />
      <CompositeNumberInput
        value={weight}
        onChange={onChange}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        maxW={20}
        defaultValue={CONSTRAINTS.initial}
      />
    </FormControl>
  );
});

Weight.displayName = 'Weight';
