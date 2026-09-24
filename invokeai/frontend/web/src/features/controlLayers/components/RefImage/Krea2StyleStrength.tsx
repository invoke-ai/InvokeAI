import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

// Krea-2's style reference has no adapter model and no begin/end step range - style strength is the one
// knob. It is a master control: besides mixing the styled attention it also pulls the reference key's
// frequency scaling and the AdaIN strength back toward neutral. At 0 the graph builder omits the style
// node altogether, so the bypass is free rather than merely invisible.
const CONSTRAINTS = {
  initial: 1,
  min: 0,
  max: 2,
  fineStep: 0.01,
  coarseStep: 0.05,
};

type Props = {
  styleStrength: number;
  onChange: (styleStrength: number) => void;
};

const formatValue = (v: number) => v.toFixed(2);
const marks = [0, 1, 2];

export const Krea2StyleStrength = memo(({ styleStrength, onChange }: Props) => {
  const { t } = useTranslation();

  return (
    <FormControl orientation="horizontal">
      <FormLabel m={0}>{t('controlLayers.krea2StyleStrength')}</FormLabel>
      <CompositeSlider
        value={styleStrength}
        onChange={onChange}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.min}
        max={CONSTRAINTS.max}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        marks={marks}
        formatValue={formatValue}
      />
      <CompositeNumberInput
        value={styleStrength}
        onChange={onChange}
        min={CONSTRAINTS.min}
        max={CONSTRAINTS.max}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        maxW={20}
        defaultValue={CONSTRAINTS.initial}
      />
    </FormControl>
  );
});

Krea2StyleStrength.displayName = 'Krea2StyleStrength';
