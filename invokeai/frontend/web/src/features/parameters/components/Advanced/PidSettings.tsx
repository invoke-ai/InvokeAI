import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  gemma2EncoderModelSelected,
  pidDecoderModelSelected,
  pidModeChanged,
  selectGemma2EncoderModel,
  selectPidDecoderModel,
  selectPidMode,
} from 'features/controlLayers/store/paramsSlice';
import type { PidMode } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGemma2EncoderModels, usePiDDecoderModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig } from 'services/api/types';

const ParamPidDecoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedModel = useAppSelector(selectPidDecoderModel);
  // PiD decoders are pinned to a backbone; only FLUX-base decoders are valid for the FLUX graph.
  const fluxOnly = useCallback((config: AnyModelConfig) => config.base === 'flux', []);
  const [modelConfigs, { isLoading }] = usePiDDecoderModels(fluxOnly);

  const _onChange = useCallback(
    (config: AnyModelConfig | null) => {
      if (config) {
        dispatch(pidDecoderModelSelected(zModelIdentifierField.parse(config)));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel,
    isLoading,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.pidDecoder')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} noOptionsMessage={noOptionsMessage} />
    </FormControl>
  );
});
ParamPidDecoderModelSelect.displayName = 'ParamPidDecoderModelSelect';

const ParamGemma2EncoderModelSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const selectedModel = useAppSelector(selectGemma2EncoderModel);
  const [modelConfigs, { isLoading }] = useGemma2EncoderModels();

  const _onChange = useCallback(
    (config: AnyModelConfig | null) => {
      if (config) {
        dispatch(gemma2EncoderModelSelected(zModelIdentifierField.parse(config)));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel,
    isLoading,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.gemma2Encoder')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} noOptionsMessage={noOptionsMessage} />
    </FormControl>
  );
});
ParamGemma2EncoderModelSelect.displayName = 'ParamGemma2EncoderModelSelect';

const PidSettings = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const pidMode = useAppSelector(selectPidMode);

  const options = useMemo<ComboboxOption[]>(
    () => [
      { value: 'off', label: t('modelManager.pidModeOff') },
      { value: 'fit', label: t('modelManager.pidModeFit') },
      { value: 'native', label: t('modelManager.pidModeNative') },
    ],
    [t]
  );

  const value = useMemo(() => options.find((o) => o.value === pidMode) ?? null, [options, pidMode]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (v) {
        dispatch(pidModeChanged(v.value as PidMode));
      }
    },
    [dispatch]
  );

  return (
    <Flex gap={4} w="full" flexDir="column">
      <FormControl gap={2}>
        <InformationalPopover feature="pidMode">
          <FormLabel m={0}>{t('modelManager.pidMode')}</FormLabel>
        </InformationalPopover>
        <Combobox value={value} options={options} onChange={onChange} />
      </FormControl>
      {pidMode !== 'off' && (
        <>
          <ParamPidDecoderModelSelect />
          <ParamGemma2EncoderModelSelect />
        </>
      )}
    </Flex>
  );
};

export default memo(PidSettings);
