import type { ComboboxOption } from '@invoke-ai/ui-library';
import { Box, Combobox, Flex, FormControl, FormLabel, Image, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { GroupBase, SingleValueProps } from 'chakra-react-select';
import { chakraComponents } from 'chakra-react-select';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectModelKey } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { modelSelected } from 'features/parameters/store/actions';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useMainModels } from 'services/api/hooks/modelsByType';
import { type AnyModelConfig, isNonRefinerMainModelConfig, type MainModelConfig } from 'services/api/types';
const commercialLicenseIcon = 'assets/images/commercial-license-icon.svg';

const ParamMainModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const activeTabName = useAppSelector(selectActiveTab);
  const selectedModelKey = useAppSelector(selectModelKey);
  const [modelConfigs, { isLoading }] = useMainModels();

  const selectedModelConfig = useMemo(() => {
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

  const shouldSelectedModelShowCommercialLicenseNotice = useMemo(
    () =>
      selectedModelConfig && isNonRefinerMainModelConfig(selectedModelConfig) && selectedModelConfig.variant === 'dev',
    [selectedModelConfig]
  );

  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    selectedModel: selectedModelConfig,
    onChange: _onChange,
    isLoading,
    getIsDisabled,
  });

  const SingleValue = ({ children, ...props }: SingleValueProps<ComboboxOption, false, GroupBase<ComboboxOption>>) => (
    <chakraComponents.SingleValue {...props}>
      <Flex gap={2}>
        {shouldSelectedModelShowCommercialLicenseNotice && (
          <Image src={commercialLicenseIcon} w={6} h={6} minW={6} minH={6} flexShrink={0} />
        )}
        {children}
      </Flex>
    </chakraComponents.SingleValue>
  );

  return (
    <FormControl isDisabled={!modelConfigs.length} isInvalid={!value || !modelConfigs.length}>
      <InformationalPopover feature="paramModel">
        <FormLabel>{t('modelManager.model')}</FormLabel>
      </InformationalPopover>
      {shouldSelectedModelShowCommercialLicenseNotice ? (
        <InformationalPopover feature="fluxDev" hideDisable={true}>
          <Box w="full" minW={0}>
            <Combobox
              menuIsOpen={true}
              value={value}
              placeholder={placeholder}
              options={options}
              onChange={onChange}
              noOptionsMessage={noOptionsMessage}
              isInvalid={value?.isDisabled}
              components={{ SingleValue }}
            />
          </Box>
        </InformationalPopover>
      ) : (
        <Tooltip label={selectedModelConfig?.description}>
          <Box w="full" minW={0}>
            <Combobox
              menuIsOpen={true}
              value={value}
              placeholder={placeholder}
              options={options}
              onChange={onChange}
              noOptionsMessage={noOptionsMessage}
              isInvalid={value?.isDisabled}
              components={{ SingleValue }}
            />
          </Box>
        </Tooltip>
      )}
    </FormControl>
  );
};

export default memo(ParamMainModelSelect);
