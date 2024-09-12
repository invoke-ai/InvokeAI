import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { selectT5EncoderModel, t5EncoderModelSelected } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useT5EncoderModels } from 'services/api/hooks/modelsByType';
import type { T5EncoderBnbQuantizedLlmInt8bModelConfig, T5EncoderModelConfig } from 'services/api/types';

const ParamT5EncoderModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const t5EncoderModel = useAppSelector(selectT5EncoderModel);
  const [modelConfigs, { isLoading }] = useT5EncoderModels();

  const _onChange = useCallback(
    (t5EncoderModel: T5EncoderBnbQuantizedLlmInt8bModelConfig | T5EncoderModelConfig | null) => {
      if (t5EncoderModel) {
        dispatch(t5EncoderModelSelected(zModelIdentifierField.parse(t5EncoderModel)));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: t5EncoderModel,
    isLoading,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} minW={0} flexGrow={1}>
      <FormLabel m={0}>{t('modelManager.t5Encoder')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} noOptionsMessage={noOptionsMessage} />
    </FormControl>
  );
};

export default memo(ParamT5EncoderModelSelect);
