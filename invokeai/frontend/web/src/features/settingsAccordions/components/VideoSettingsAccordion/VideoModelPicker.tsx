import { Flex, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { selectVideoModel, videoModelChanged } from 'features/parameters/store/videoSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useVideoModels } from 'services/api/hooks/modelsByType';
import type { RunwayModelConfig, Veo3ModelConfig } from 'services/api/types';

export const VideoModelPicker = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [modelConfigs] = useVideoModels();
  const selectedModelConfig = useAppSelector(selectVideoModel);
  const onChange = useCallback(
    (modelConfig: Veo3ModelConfig | RunwayModelConfig) => {
      dispatch(videoModelChanged(modelConfig));
    },
    [dispatch]
  );

  return (
    <Flex alignItems="center" gap={2}>
      <InformationalPopover feature="paramModel">
        <FormLabel>{t('modelManager.model')}</FormLabel>
      </InformationalPopover>
      <ModelPicker
        pickerId="main-model"
        modelConfigs={modelConfigs}
        selectedModelConfig={selectedModelConfig as Veo3ModelConfig | RunwayModelConfig | undefined}
        onChange={onChange}
        grouped
      />
    </Flex>
  );
});
VideoModelPicker.displayName = 'VideoModelPicker';
