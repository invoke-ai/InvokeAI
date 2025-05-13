import { Flex, FormLabel, Icon } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { useSimpleTabModelConfig } from 'features/simpleGeneration/hooks/useSimpleTabModelConfig';
import { modelChanged } from 'features/simpleGeneration/store/simpleGenerationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdMoneyOff } from 'react-icons/md';
import { useSimpleTabModels } from 'services/api/hooks/modelsByType';
import { type AnyModelConfig, isCheckpointMainModelConfig } from 'services/api/types';

export const SimpleTabModel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [modelConfigs] = useSimpleTabModels();
  const selectedModelConfig = useSimpleTabModelConfig();
  const onChange = useCallback(
    (modelConfig: AnyModelConfig) => {
      dispatch(modelChanged({ model: zModelIdentifierField.parse(modelConfig) }));
    },
    [dispatch]
  );

  const isFluxDevSelected = useMemo(
    () =>
      selectedModelConfig &&
      isCheckpointMainModelConfig(selectedModelConfig) &&
      selectedModelConfig.config_path === 'flux-dev',
    [selectedModelConfig]
  );

  return (
    <Flex alignItems="center" gap={2}>
      <InformationalPopover feature="paramModel">
        <FormLabel>{t('modelManager.model')}</FormLabel>
      </InformationalPopover>
      {isFluxDevSelected && (
        <InformationalPopover feature="fluxDevLicense" hideDisable={true}>
          <Flex justifyContent="flex-start">
            <Icon as={MdMoneyOff} />
          </Flex>
        </InformationalPopover>
      )}
      <ModelPicker modelConfigs={modelConfigs} selectedModelConfig={selectedModelConfig} onChange={onChange} grouped />
    </Flex>
  );
});
SimpleTabModel.displayName = 'SimpleTabModel';
