import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { selectTileControlNetModel, tileControlnetModelChanged } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useControlNetModels } from 'services/api/hooks/modelsByType';
import type { ControlNetModelConfig } from 'services/api/types';

const ParamTileControlNetModel = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const tileControlNetModel = useAppSelector(selectTileControlNetModel);
  const [modelConfigs, { isLoading }] = useControlNetModels();

  const _onChange = useCallback(
    (controlNetModel: ControlNetModelConfig | null) => {
      dispatch(tileControlnetModelChanged(controlNetModel));
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: tileControlNetModel,
    isLoading,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('controlNet')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} noOptionsMessage={noOptionsMessage} />
    </FormControl>
  );
};

export default memo(ParamTileControlNetModel);