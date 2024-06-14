import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type {ProcessorConfig } from 'features/controlLayers/store/types';
import { CA_PROCESSOR_DATA, isProcessorTypeV2 } from 'features/controlLayers/store/types';
import { configSelector } from 'features/system/store/configSelectors';
import { includes, map } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = {
  config: ProcessorConfig | null;
  onChange: (config: ProcessorConfig | null) => void;
};

const selectDisabledProcessors = createMemoizedSelector(
  configSelector,
  (config) => config.sd.disabledControlNetProcessors
);

export const CAProcessorTypeSelect = memo(({ config, onChange }: Props) => {
  const { t } = useTranslation();
  const disabledProcessors = useAppSelector(selectDisabledProcessors);
  const options = useMemo(() => {
    return map(CA_PROCESSOR_DATA, ({ labelTKey }, type) => ({ value: type, label: t(labelTKey) })).filter(
      (o) => !includes(disabledProcessors, o.value)
    );
  }, [disabledProcessors, t]);

  const _onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        onChange(null);
      } else {
        assert(isProcessorTypeV2(v.value));
        onChange(CA_PROCESSOR_DATA[v.value].buildDefaults());
      }
    },
    [onChange]
  );
  const clearProcessor = useCallback(() => {
    onChange(null);
  }, [onChange]);
  const value = useMemo(() => options.find((o) => o.value === config?.type) ?? null, [options, config?.type]);

  return (
    <Flex gap={2}>
      <FormControl>
        <InformationalPopover feature="controlNetProcessor">
          <FormLabel m={0}>{t('controlnet.processor')}</FormLabel>
        </InformationalPopover>
        <Combobox value={value} options={options} onChange={_onChange} isSearchable={false} isClearable={false} />
      </FormControl>
      <IconButton
        aria-label={t('controlLayers.clearProcessor')}
        onClick={clearProcessor}
        isDisabled={!config}
        icon={<PiXBold />}
        variant="ghost"
        size="sm"
      />
    </Flex>
  );
});

CAProcessorTypeSelect.displayName = 'CAProcessorTypeSelect';
