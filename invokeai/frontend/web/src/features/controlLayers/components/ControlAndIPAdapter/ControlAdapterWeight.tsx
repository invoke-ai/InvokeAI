import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  weight: number;
  onChange: (weight: number) => void;
};

const formatValue = (v: number) => v.toFixed(2);
const marks = [0, 1, 2];

export const ControlAdapterWeight = memo(({ weight, onChange }: Props) => {
  const { t } = useTranslation();
  const initial = useAppSelector((s) => s.config.sd.ca.weight.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.ca.weight.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.ca.weight.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.ca.weight.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.ca.weight.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.ca.weight.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.ca.weight.fineStep);

  return (
    <FormControl orientation="horizontal">
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
});

ControlAdapterWeight.displayName = 'ControlAdapterWeight';
