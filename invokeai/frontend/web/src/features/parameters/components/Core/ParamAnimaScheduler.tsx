import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectAnimaScheduler, setAnimaScheduler } from 'features/controlLayers/store/paramsSlice';
import { isParameterAnimaScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

// Anima scheduler options (same flow-matching schedulers as Z-Image)
const ANIMA_SCHEDULER_OPTIONS: ComboboxOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'heun', label: 'Heun (2nd order)' },
  { value: 'lcm', label: 'LCM' },
];

const ParamAnimaScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const animaScheduler = useAppSelector(selectAnimaScheduler);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      // Validate against Anima scheduler values
      if (!isParameterAnimaScheduler(v?.value)) {
        return;
      }
      dispatch(setAnimaScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => ANIMA_SCHEDULER_OPTIONS.find((o) => o.value === animaScheduler), [animaScheduler]);

  return (
    <FormControl>
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={ANIMA_SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamAnimaScheduler);
