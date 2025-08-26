import { Flex, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { videoModelChanged } from 'features/parameters/store/videoSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useVideoModels } from 'services/api/hooks/modelsByType';
import { useSelectedVideoModelConfig } from 'services/api/hooks/useSelectedModelConfig';
import type { VideoApiModelConfig } from 'services/api/types';

export const VideoModelPicker = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [modelConfigs] = useVideoModels();
  const selectedModelConfig = useSelectedVideoModelConfig();
  const onChange = useCallback(
    (modelConfig: VideoApiModelConfig) => {
      dispatch(videoModelChanged({ videoModel: zModelIdentifierField.parse(modelConfig) }));
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
        selectedModelConfig={selectedModelConfig}
        onChange={onChange}
        grouped
      />
    </Flex>
  );
});
VideoModelPicker.displayName = 'VideoModelPicker';
