import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectMainModelConfig,
  selectZImageScheduler,
  setZImageScheduler,
} from 'features/controlLayers/store/paramsSlice';
import { isParameterZImageScheduler } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

// All Z-Image scheduler options
const ZIMAGE_SCHEDULER_OPTIONS_ALL: ComboboxOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'heun', label: 'Heun (2nd order)' },
  { value: 'lcm', label: 'LCM' },
];

// Z-Image Base (zbase) scheduler options - LCM not supported for undistilled models
const ZIMAGE_SCHEDULER_OPTIONS_BASE: ComboboxOption[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'heun', label: 'Heun (2nd order)' },
];

const ParamZImageScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const zImageScheduler = useAppSelector(selectZImageScheduler);
  const modelConfig = useAppSelector(selectMainModelConfig);

  // Determine if the selected model is Z-Image Base (zbase variant)
  // LCM is not supported for undistilled models
  // Need to check base first to narrow the type, since only z-image models have this variant
  const isZImageBase = modelConfig?.base === 'z-image' && modelConfig.variant === 'zbase';
  const options = isZImageBase ? ZIMAGE_SCHEDULER_OPTIONS_BASE : ZIMAGE_SCHEDULER_OPTIONS_ALL;

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterZImageScheduler(v?.value)) {
        return;
      }
      dispatch(setZImageScheduler(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === zImageScheduler), [options, zImageScheduler]);

  return (
    <FormControl>
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamZImageScheduler);
