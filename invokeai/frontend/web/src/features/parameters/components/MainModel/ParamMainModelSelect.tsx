import { Box, Combobox, FormControl, FormLabel, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { modelSelected } from 'features/parameters/store/actions';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useMainModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig, MainModelConfig } from 'services/api/types';

const ParamMainModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const activeTabName = useAppSelector(selectActiveTab);
  const selectedModel = useAppSelector(selectModel);
  const [modelConfigs, { isLoading }] = useMainModels();

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

  return (
    <FormControl isDisabled={!modelConfigs.length} isInvalid={!value || !modelConfigs.length}>
      <InformationalPopover feature="paramModel">
        <FormLabel>{t('modelManager.model')}</FormLabel>
      </InformationalPopover>
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
    </FormControl>
  );
};

export default memo(ParamMainModelSelect);
