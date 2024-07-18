import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { upscaleModelChanged } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSpandrelImageToImageModels } from 'services/api/hooks/modelsByType';
import type { SpandrelImageToImageModelConfig } from 'services/api/types';

const ParamSpandrelModel = () => {
  const { t } = useTranslation();
  const [modelConfigs, { isLoading }] = useSpandrelImageToImageModels();

  const model = useAppSelector((s) => s.upscale.upscaleModel);

  const dispatch = useAppDispatch();

  const _onChange = useCallback(
    (v: SpandrelImageToImageModelConfig | null) => {
      dispatch(upscaleModelChanged(v));
    },
    [dispatch]
  );

  const { options, value, onChange, placeholder, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: model,
    isLoading,
  });

  return (
    <FormControl orientation="vertical">
      <FormLabel>Upscale Model</FormLabel>
      <Combobox
        value={value}
        placeholder={placeholder}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
      />
    </FormControl>
  );
};

export default memo(ParamSpandrelModel);
