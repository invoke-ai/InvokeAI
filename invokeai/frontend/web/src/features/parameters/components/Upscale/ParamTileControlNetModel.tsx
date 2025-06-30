import { FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectBase } from 'features/controlLayers/store/paramsSlice';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { selectTileControlNetModel, tileControlnetModelChanged } from 'features/parameters/store/upscaleSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useControlNetModels } from 'services/api/hooks/modelsByType';
import type { ControlNetModelConfig } from 'services/api/types';

const ParamTileControlNetModel = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const tileControlNetModel = useAppSelector(selectTileControlNetModel);
  const currentBaseModel = useAppSelector(selectBase);
  const [modelConfigs, { isLoading }] = useControlNetModels();

  const _onChange = useCallback(
    (controlNetModel: ControlNetModelConfig) => {
      dispatch(tileControlnetModelChanged(controlNetModel));
    },
    [dispatch]
  );

  const filteredModelConfigs = useMemo(() => {
    if (!currentBaseModel) {
      return [];
    }
    return modelConfigs.filter((model) => {
      const isCompatible = model.base === currentBaseModel;
      const isTileOrMultiModel = model.name.toLowerCase().includes('tile') || model.name.toLowerCase().includes('union');
      return isCompatible && isTileOrMultiModel;
    });
  }, [modelConfigs, currentBaseModel]);

  const getIsOptionDisabled = useCallback(
    (model: ControlNetModelConfig): boolean => {
      const isCompatible = currentBaseModel === model.base;
      const hasMainModel = Boolean(currentBaseModel);
      return !hasMainModel || !isCompatible;
    },
    [currentBaseModel]
  );

  return (
    <FormControl isDisabled={!filteredModelConfigs.length} isInvalid={!filteredModelConfigs.length} minW={0} flexGrow={1} gap={2}>
      <InformationalPopover feature="controlNet">
        <FormLabel m={0}>{t('upscaling.tileControl')}</FormLabel>
      </InformationalPopover>
      <ModelPicker
        modelConfigs={filteredModelConfigs}
        selectedModelConfig={tileControlNetModel ?? undefined}
        onChange={_onChange}
        getIsOptionDisabled={getIsOptionDisabled}
        placeholder={t('common.placeholderSelectAModel')}
        noOptionsText={t('upscaling.missingTileControlNetModel')}
        isDisabled={isLoading || !filteredModelConfigs.length}
        isInvalid={!filteredModelConfigs.length}
      />
    </FormControl>
  );
};

export default memo(ParamTileControlNetModel);