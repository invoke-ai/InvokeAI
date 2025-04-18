import { Box, Combobox, Flex, FormControl, FormLabel, Icon, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectModelKey } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import { UseDefaultSettingsButton } from 'features/parameters/components/MainModel/UseDefaultSettingsButton';
import { modelSelected } from 'features/parameters/store/actions';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdMoneyOff } from 'react-icons/md';
import { useMainModels } from 'services/api/hooks/modelsByType';
import { type AnyModelConfig, isCheckpointMainModelConfig, type MainModelConfig } from 'services/api/types';

const ParamMainModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const activeTabName = useAppSelector(selectActiveTab);
  const selectedModelKey = useAppSelector(selectModelKey);
  // const selectedModel = useAppSelector(selectModel);
  const [modelConfigs, { isLoading }] = useMainModels();

  const selectedModel = useMemo(() => {
    if (!modelConfigs) {
      return null;
    }
    if (selectedModelKey === null) {
      return null;
    }
    const modelConfig = modelConfigs.find((model) => model.key === selectedModelKey);

    if (!modelConfig) {
      return null;
    }

    return modelConfig;
  }, [modelConfigs, selectedModelKey]);

  const tooltipLabel = useMemo(() => {
    if (!modelConfigs.length || !selectedModel) {
      return;
    }
    return modelConfigs.find((m) => m.key === selectedModel?.key)?.description;
  }, [modelConfigs, selectedModel]);

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (!model) {
        return;
      }
      try {
        dispatch(modelSelected(zModelIdentifierField.parse(model)));
      } catch {
        // no-op
      }
    },
    [dispatch]
  );

  const getIsDisabled = useCallback(
    (model: AnyModelConfig): boolean => {
      return activeTabName === 'upscaling' && model.base === 'flux';
    },
    [activeTabName]
  );

  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    selectedModel,
    onChange: _onChange,
    isLoading,
    getIsDisabled,
  });

  const isFluxDevSelected = useMemo(() => {
    return selectedModel && isCheckpointMainModelConfig(selectedModel) && selectedModel.config_path === 'flux-dev';
  }, [selectedModel]);

  return (
    <FormControl isDisabled={!modelConfigs.length} isInvalid={!value || !modelConfigs.length} gap={2}>
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
      <Tooltip label={tooltipLabel}>
        <Box w="full" minW={0}>
          <Combobox
            value={value}
            placeholder={placeholder}
            options={options}
            onChange={onChange}
            noOptionsMessage={noOptionsMessage}
            isInvalid={value?.isDisabled}
          />
        </Box>
      </Tooltip>
      <NavigateToModelManagerButton />
      <UseDefaultSettingsButton />
    </FormControl>
  );
};

export default memo(ParamMainModelSelect);
