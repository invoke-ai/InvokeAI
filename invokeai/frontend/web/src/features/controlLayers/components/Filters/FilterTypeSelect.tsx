import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { map } from 'es-toolkit/compat';
import type { FilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS, isFilterType } from 'features/controlLayers/store/filters';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

type Props = {
  filterType: FilterConfig['type'];
  onChange: (filterType: FilterConfig['type']) => void;
};

export const FilterTypeSelect = memo(({ filterType, onChange }: Props) => {
  const { t } = useTranslation();
  const options = useMemo(() => {
    return map(IMAGE_FILTERS, (data, type) => ({ value: type, label: t(`controlLayers.filter.${type}.label`) }));
  }, [t]);

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
