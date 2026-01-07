import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectFluxScheduler, setFluxScheduler } from 'features/controlLayers/store/paramsSlice';
import { isParameterFluxScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

// Flux-specific scheduler options (Flow Matching schedulers)
const FLUX_SCHEDULER_OPTIONS: ComboboxOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'heun', label: 'Heun (2nd order)' },
  { value: 'lcm', label: 'LCM' },
];

const ParamFluxScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const fluxScheduler = useAppSelector(selectFluxScheduler);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterFluxScheduler(v?.value)) {
        return;
      }
      dispatch(setFluxScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => FLUX_SCHEDULER_OPTIONS.find((o) => o.value === fluxScheduler), [fluxScheduler]);

  return (
    <FormControl>
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={FLUX_SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamFluxScheduler);
