import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectErnieImageScheduler, setErnieImageScheduler } from 'features/controlLayers/store/paramsSlice';
import { isParameterErnieImageScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ERNIE_IMAGE_SCHEDULER_OPTIONS: ComboboxOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'heun', label: 'Heun (2nd order)' },
  { value: 'lcm', label: 'LCM' },
];

const ParamErnieImageScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const ernieImageScheduler = useAppSelector(selectErnieImageScheduler);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterErnieImageScheduler(v?.value)) {
        return;
      }
      dispatch(setErnieImageScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => ERNIE_IMAGE_SCHEDULER_OPTIONS.find((o) => o.value === ernieImageScheduler),
    [ernieImageScheduler]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={ERNIE_IMAGE_SCHEDULER_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamErnieImageScheduler);
