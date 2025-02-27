import { CompositeRangeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  beginEndStepPct: [number, number];
  onChange: (beginEndStepPct: [number, number]) => void;
};

const formatPct = (v: number) => `${Math.round(v * 100)}%`;
const ariaLabel = ['Begin Step %', 'End Step %'];

export const BeginEndStepPct = memo(({ beginEndStepPct, onChange }: Props) => {
  const { t } = useTranslation();
  const onReset = useCallback(() => {
    onChange([0, 1]);
  }, [onChange]);

  return (
    <FormControl orientation="horizontal" pe={2}>
      <InformationalPopover feature="controlNetBeginEnd">
        <FormLabel m={0}>{t('controlLayers.beginEndStepPercentShort')}</FormLabel>
      </InformationalPopover>
      <CompositeRangeSlider
        aria-label={ariaLabel}
        value={beginEndStepPct}
        onChange={onChange}
        onReset={onReset}
        min={0}
        max={1}
        step={0.05}
        fineStep={0.01}
        minStepsBetweenThumbs={1}
        formatValue={formatPct}
        marks
        withThumbTooltip
      />
    </FormControl>
  );
});

BeginEndStepPct.displayName = 'BeginEndStepPct';
