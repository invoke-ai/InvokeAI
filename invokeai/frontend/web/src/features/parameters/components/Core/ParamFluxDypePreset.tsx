import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectFluxDypePreset, setFluxDypePreset } from 'features/controlLayers/store/paramsSlice';
import { isParameterFluxDypePreset } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

// DyPE (Dynamic Position Extrapolation) preset options for high-resolution generation
const FLUX_DYPE_PRESET_OPTIONS: ComboboxOption[] = [
  { value: 'off', label: 'Off' },
  { value: 'auto', label: 'Auto (> 1536px)' },
  { value: '4k', label: '4K Optimized' },
];

const ParamFluxDypePreset = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const fluxDypePreset = useAppSelector(selectFluxDypePreset);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterFluxDypePreset(v?.value)) {
        return;
      }
      dispatch(setFluxDypePreset(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => FLUX_DYPE_PRESET_OPTIONS.find((o) => o.value === fluxDypePreset), [fluxDypePreset]);

  return (
    <FormControl>
      <InformationalPopover feature="fluxDypePreset">
        <FormLabel>{t('parameters.dypePreset')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={FLUX_DYPE_PRESET_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamFluxDypePreset);
