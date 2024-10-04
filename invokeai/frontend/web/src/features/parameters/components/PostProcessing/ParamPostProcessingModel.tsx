import { Box, Combobox, FormControl, FormLabel, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { postProcessingModelChanged, selectPostProcessingModel } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useSpandrelImageToImageModels } from 'services/api/hooks/modelsByType';
import type { SpandrelImageToImageModelConfig } from 'services/api/types';

const ParamPostProcessingModel = () => {
  const { t } = useTranslation();
  const [modelConfigs, { isLoading }] = useSpandrelImageToImageModels();

  const model = useAppSelector(selectPostProcessingModel);
  const dispatch = useAppDispatch();

  const tooltipLabel = useMemo(() => {
    if (!modelConfigs.length || !model) {
      return;
    }
    return modelConfigs.find((m) => m.key === model?.key)?.description;
  }, [modelConfigs, model]);

  const _onChange = useCallback(
    (v: SpandrelImageToImageModelConfig | null) => {
      dispatch(postProcessingModelChanged(v));
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
      <FormLabel>{t('upscaling.postProcessingModel')}</FormLabel>
      <Tooltip label={tooltipLabel}>
        <Box w="full">
          <Combobox
            value={value}
            placeholder={placeholder}
            options={options}
            onChange={onChange}
            noOptionsMessage={noOptionsMessage}
            isDisabled={options.length === 0}
          />
        </Box>
      </Tooltip>
    </FormControl>
  );
};

export default memo(ParamPostProcessingModel);
