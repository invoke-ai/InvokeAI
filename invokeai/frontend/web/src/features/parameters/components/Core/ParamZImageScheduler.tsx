import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectZImageScheduler, setZImageScheduler } from 'features/controlLayers/store/paramsSlice';
import { isParameterZImageScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

// Z-Image scheduler options (Flow Matching schedulers, same as Flux)
const ZIMAGE_SCHEDULER_OPTIONS: ComboboxOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'heun', label: 'Heun (2nd order)' },
  { value: 'lcm', label: 'LCM' },
];

const ParamZImageScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const zImageScheduler = useAppSelector(selectZImageScheduler);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterZImageScheduler(v?.value)) {
        return;
      }
      dispatch(setZImageScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => ZIMAGE_SCHEDULER_OPTIONS.find((o) => o.value === zImageScheduler), [zImageScheduler]);

  return (
    <FormControl>
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={ZIMAGE_SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamZImageScheduler);
