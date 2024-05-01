import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { caLayerProcessorConfigChanged, selectCALayer } from 'features/controlLayers/store/controlLayersSlice';
import { CONTROLNET_PROCESSORS, isProcessorType } from 'features/controlLayers/util/controlAdapters';
import { configSelector } from 'features/system/store/configSelectors';
import { includes, map } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

const selectDisabledProcessors = createMemoizedSelector(
  configSelector,
  (config) => config.sd.disabledControlNetProcessors
);

export const CALayerProcessorCombobox = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const disabledProcessors = useAppSelector(selectDisabledProcessors);
  const processorType = useAppSelector(
    (s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter.processorConfig?.type ?? null
  );
  const options = useMemo(() => {
    return map(CONTROLNET_PROCESSORS, ({ labelTKey }, type) => ({ value: type, label: t(labelTKey) })).filter(
      (o) => !includes(disabledProcessors, o.value)
    );
  }, [disabledProcessors, t]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        dispatch(
          caLayerProcessorConfigChanged({
            layerId,
            processorConfig: null,
          })
        );
        return;
      }
      assert(isProcessorType(v.value));
      dispatch(
        caLayerProcessorConfigChanged({
          layerId,
          processorConfig: CONTROLNET_PROCESSORS[v.value].buildDefaults(),
        })
      );
    },
    [dispatch, layerId]
  );
  const clearProcessor = useCallback(() => {
    dispatch(
      caLayerProcessorConfigChanged({
        layerId,
        processorConfig: null,
      })
    );
  }, [dispatch, layerId]);
  const value = useMemo(() => options.find((o) => o.value === processorType) ?? null, [options, processorType]);

  return (
    <FormControl>
      <InformationalPopover feature="controlNetProcessor">
        <FormLabel>{t('controlnet.processor')}</FormLabel>
      </InformationalPopover>
      <Flex gap={4}>
        <Combobox value={value} options={options} onChange={onChange} />
        <IconButton
          aria-label={t('controlnet.processor')}
          onClick={clearProcessor}
          icon={<PiXBold />}
          variant="ghost"
        />
      </Flex>
    </FormControl>
  );
});

CALayerProcessorCombobox.displayName = 'CALayerProcessorCombobox';
