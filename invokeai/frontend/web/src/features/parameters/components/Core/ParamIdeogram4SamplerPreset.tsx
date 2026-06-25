import type { ComboboxOnChange, ComboboxOption, SystemStyleObject } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectIdeogram4SamplerPreset, setIdeogram4SamplerPreset } from 'features/controlLayers/store/paramsSlice';
import { isParameterIdeogram4SamplerPreset } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

// Each preset bundles a step count, the per-step guidance schedule (with a polish tail), and the
// logit-normal schedule mean/std. The primary quality/speed control for Ideogram 4.
const IDEOGRAM4_SAMPLER_PRESET_OPTIONS: ComboboxOption[] = [
  { value: 'V4_QUALITY_48', label: 'Quality (48 steps)' },
  { value: 'V4_DEFAULT_20', label: 'Default (20 steps)' },
  { value: 'V4_TURBO_12', label: 'Turbo (12 steps)' },
];

// Per-preset step count and schedule mean (mu), mirroring the backend PRESETS. Used by the advanced
// override controls to show the active preset's value as the "auto" default.
export const IDEOGRAM4_PRESET_DEFAULTS: Record<string, { steps: number; mu: number }> = {
  V4_QUALITY_48: { steps: 48, mu: 0.0 },
  V4_DEFAULT_20: { steps: 20, mu: 0.0 },
  V4_TURBO_12: { steps: 12, mu: 0.5 },
};

// Cap the width so the dropdown doesn't span the whole row, and push it to the right edge of the row
// (ms: 'auto') so it lines up with the other controls' right edge. The label keeps its normal position.
const comboboxSx: SystemStyleObject = { maxW: '13rem', ms: 'auto' };

const ParamIdeogram4SamplerPreset = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const samplerPreset = useAppSelector(selectIdeogram4SamplerPreset);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterIdeogram4SamplerPreset(v?.value)) {
        return;
      }
      dispatch(setIdeogram4SamplerPreset(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => IDEOGRAM4_SAMPLER_PRESET_OPTIONS.find((o) => o.value === samplerPreset), [samplerPreset]);

  return (
    <FormControl>
      <FormLabel>{t('parameters.samplerPreset')}</FormLabel>
      <Combobox value={value} options={IDEOGRAM4_SAMPLER_PRESET_OPTIONS} onChange={onChange} sx={comboboxSx} />
    </FormControl>
  );
};

export default memo(ParamIdeogram4SamplerPreset);
