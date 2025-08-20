import { Flex, FormLabel, Icon } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { modelSelected } from 'features/parameters/store/actions';
import { selectVideoModel, videoModelChanged } from 'features/parameters/store/videoSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdMoneyOff } from 'react-icons/md';
import { useMainModels, useVeo3Models } from 'services/api/hooks/modelsByType';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';
import { type AnyModelConfig, isCheckpointMainModelConfig, Veo3ModelConfig } from 'services/api/types';

export const VideoModelPicker = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [modelConfigs] = useVeo3Models();
  const selectedModelConfig = useAppSelector(selectVideoModel);
  const onChange = useCallback(
    (modelConfig: Veo3ModelConfig) => {
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
        selectedModelConfig={selectedModelConfig as Veo3ModelConfig | undefined}
        onChange={onChange}
        grouped
      />
    </Flex>
  );
});
VideoModelPicker.displayName = 'VideoModelPicker';
