import { Flex, FormControl, FormLabel, Icon } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { modelSelected } from 'features/parameters/store/actions';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdMoneyOff } from 'react-icons/md';
import { useMainModels } from 'services/api/hooks/modelsByType';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';
import { type AnyModelConfig, isCheckpointMainModelConfig } from 'services/api/types';

export const InitialStateMainModelPicker = memo(() => {
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
    <FormControl orientation="vertical" alignItems="unset">
      <FormLabel display="flex" fontSize="md" gap={2}>
        {t('common.selectYourModel')}{' '}
        {isFluxDevSelected && (
          <InformationalPopover feature="fluxDevLicense" hideDisable={true}>
            <Flex justifyContent="flex-start">
              <Icon as={MdMoneyOff} />
            </Flex>
          </InformationalPopover>
        )}
      </FormLabel>
      <ModelPicker
        pickerId="initial-state-main-model"
        modelConfigs={modelConfigs}
        selectedModelConfig={selectedModelConfig}
        onChange={onChange}
        grouped
      />
    </FormControl>
  );
});
InitialStateMainModelPicker.displayName = 'InitialStateMainModelPicker';
