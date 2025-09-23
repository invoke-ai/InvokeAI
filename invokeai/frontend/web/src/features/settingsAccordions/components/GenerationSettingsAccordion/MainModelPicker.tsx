import { Flex, FormLabel, Icon } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { modelSelected } from 'features/parameters/store/actions';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdMoneyOff } from 'react-icons/md';
import { useMainModels } from 'services/api/hooks/modelsByType';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';
import { type AnyModelConfig, isCheckpointMainModelConfig } from 'services/api/types';

export const MainModelPicker = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [modelConfigs] = useMainModels();
  const selectedModelConfig = useSelectedModelConfig();
  const onChange = useCallback(
    (modelConfig: AnyModelConfig) => {
      dispatch(modelSelected(modelConfig));
    },
    [dispatch]
  );

  const isFluxDevSelected = useMemo(
    () =>
      selectedModelConfig &&
      isCheckpointMainModelConfig(selectedModelConfig) &&
      selectedModelConfig.variant === 'flux_dev',
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
      <ModelPicker
        pickerId="main-model"
        modelConfigs={modelConfigs}
        selectedModelConfig={selectedModelConfig}
        onChange={onChange}
        grouped
      />
      <UseDefaultSettingsButton />
    </Flex>
  );
});
MainModelPicker.displayName = 'MainModelPicker';
