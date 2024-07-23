import { Box, Combobox, FormControl, FormLabel, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { simpleUpscaleModelChanged, upscaleModelChanged } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useSpandrelImageToImageModels } from 'services/api/hooks/modelsByType';
import type { SpandrelImageToImageModelConfig } from 'services/api/types';

interface Props {
  isMultidiffusion: boolean;
}

const ParamSpandrelModel = ({ isMultidiffusion }: Props) => {
  const { t } = useTranslation();
  const [modelConfigs, { isLoading }] = useSpandrelImageToImageModels();

  const model = useAppSelector((s) => isMultidiffusion ? s.upscale.upscaleModel : s.upscale.simpleUpscaleModel);
  const dispatch = useAppDispatch();

  const tooltipLabel = useMemo(() => {
    if (!modelConfigs.length || !model) {
      return;
    }
    return modelConfigs.find((m) => m.key === model?.key)?.description;
  }, [modelConfigs, model]);

  const _onChange = useCallback(
    (v: SpandrelImageToImageModelConfig | null) => {
      if (isMultidiffusion) {
        dispatch(upscaleModelChanged(v));
      } else {
        dispatch(simpleUpscaleModelChanged(v))
      }
    },
    [isMultidiffusion, dispatch]
  );

  const { options, value, onChange, placeholder, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: model,
    isLoading,
  });

  return (
    <FormControl orientation="vertical">
      <FormLabel>{t('upscaling.upscaleModel')}</FormLabel>
      <Tooltip label={tooltipLabel}>
        <Box w="full">
          <Combobox
            value={value}
            placeholder={placeholder}
            options={options}
            onChange={onChange}
            noOptionsMessage={noOptionsMessage}
          />
        </Box>
      </Tooltip>
    </FormControl>
  );
};

export default memo(ParamSpandrelModel);
