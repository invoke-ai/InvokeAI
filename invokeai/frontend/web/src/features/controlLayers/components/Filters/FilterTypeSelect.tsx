import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { FilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS, isFilterType } from 'features/controlLayers/store/filters';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { includes, map } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

const selectDisabledProcessors = createSelector(selectConfigSlice, (config) => config.sd.disabledControlNetProcessors);

type Props = {
  filterType: FilterConfig['type'];
  onChange: (filterType: FilterConfig['type']) => void;
};

export const FilterTypeSelect = memo(({ filterType, onChange }: Props) => {
  const { t } = useTranslation();
  const disabledProcessors = useAppSelector(selectDisabledProcessors);
  const options = useMemo(() => {
    return map(IMAGE_FILTERS, (data, type) => ({ value: type, label: t(`controlLayers.filter.${type}.label`) })).filter(
      (o) => !includes(disabledProcessors, o.value)
    );
  }, [disabledProcessors, t]);

  const _onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      assert(isFilterType(v.value));
      onChange(v.value);
    },
    [onChange]
  );
  const value = useMemo(() => options.find((o) => o.value === filterType) ?? null, [options, filterType]);

  return (
    <Flex gap={2}>
      <FormControl>
        <InformationalPopover feature="controlNetProcessor">
          <FormLabel m={0}>{t('controlLayers.filter.filterType')}</FormLabel>
        </InformationalPopover>
        <Combobox value={value} options={options} onChange={_onChange} isSearchable={false} isClearable={false} />
      </FormControl>
    </Flex>
  );
});

FilterTypeSelect.displayName = 'FilterTypeSelect';
