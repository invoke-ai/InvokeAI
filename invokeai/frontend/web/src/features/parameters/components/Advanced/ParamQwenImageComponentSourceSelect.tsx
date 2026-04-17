import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  qwenImageComponentSourceSelected,
  selectQwenImageComponentSource,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useQwenImageDiffusersModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig } from 'services/api/types';

/**
 * Qwen Image Edit Component Source Model Select
 *
 * Selects a Diffusers Qwen Image Edit model to provide the VAE and text encoder
 * when using a GGUF quantized transformer.
 */
const ParamQwenImageComponentSourceSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const componentSource = useAppSelector(selectQwenImageComponentSource);
  const [modelConfigs, { isLoading }] = useQwenImageDiffusersModels();

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (model) {
        dispatch(qwenImageComponentSourceSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(qwenImageComponentSourceSelected(null));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: componentSource,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.qwenImageComponentSource')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.qwenImageComponentSourcePlaceholder')}
      />
    </FormControl>
  );
});

ParamQwenImageComponentSourceSelect.displayName = 'ParamQwenImageComponentSourceSelect';

export default ParamQwenImageComponentSourceSelect;
