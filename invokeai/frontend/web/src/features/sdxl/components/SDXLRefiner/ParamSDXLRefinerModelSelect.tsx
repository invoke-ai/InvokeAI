import { Box, Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { refinerModelChanged } from 'features/controlLayers/store/canvasV2Slice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useRefinerModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig } from 'services/api/types';

const optionsFilter = (model: MainModelConfig) => model.base === 'sdxl-refiner';

const ParamSDXLRefinerModelSelect = () => {
  const dispatch = useAppDispatch();
  const model = useAppSelector((s) => s.canvasV2.params.refinerModel);
  const { t } = useTranslation();
  const [modelConfigs, { isLoading }] = useRefinerModels();
  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (!model) {
        dispatch(refinerModelChanged(null));
        return;
      }
      dispatch(refinerModelChanged(zModelIdentifierField.parse(model)));
    },
    [dispatch]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: model,
    isLoading,
    optionsFilter,
  });
  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} w="full">
      <InformationalPopover feature="refinerModel">
        <FormLabel>{t('sdxl.refinermodel')}</FormLabel>
      </InformationalPopover>
      <Box w="full" minW={0}>
        <Combobox
          value={value}
          placeholder={placeholder}
          options={options}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
          isClearable
        />
      </Box>
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerModelSelect);
