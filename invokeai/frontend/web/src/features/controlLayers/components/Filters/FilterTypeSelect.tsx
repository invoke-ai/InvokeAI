import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { filterSelected } from 'features/controlLayers/store/canvasV2Slice';
import { IMAGE_FILTERS, isFilterType } from 'features/controlLayers/store/types';
import { configSelector } from 'features/system/store/configSelectors';
import { includes, map } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

const selectDisabledProcessors = createMemoizedSelector(
  configSelector,
  (config) => config.sd.disabledControlNetProcessors
);

export const FilterTypeSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const filterType = useAppSelector((s) => s.canvasV2.filter.config.type);
  const disabledProcessors = useAppSelector(selectDisabledProcessors);
  const options = useMemo(() => {
    return map(IMAGE_FILTERS, ({ labelTKey }, type) => ({ value: type, label: t(labelTKey) })).filter(
      (o) => !includes(disabledProcessors, o.value)
    );
  }, [disabledProcessors, t]);

  const _onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      assert(isFilterType(v.value));
      dispatch(filterSelected({ type: v.value }));
    },
    [dispatch]
  );
  const value = useMemo(() => options.find((o) => o.value === filterType) ?? null, [options, filterType]);

  return (
    <Flex gap={2}>
      <FormControl>
        <InformationalPopover feature="controlNetProcessor">
          <FormLabel m={0}>{t('controlLayers.filter')}</FormLabel>
        </InformationalPopover>
        <Combobox value={value} options={options} onChange={_onChange} isSearchable={false} isClearable={false} />
      </FormControl>
    </Flex>
  );
});

FilterTypeSelect.displayName = 'FilterTypeSelect';
