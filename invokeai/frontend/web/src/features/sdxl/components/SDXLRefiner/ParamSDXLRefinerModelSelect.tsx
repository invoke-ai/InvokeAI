import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { refinerModelChanged, selectSdxlSlice } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { REFINER_BASE_MODELS } from 'services/api/constants';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';
import type { MainModelConfig } from 'services/api/types';

const selectModel = createMemoizedSelector(selectSdxlSlice, (sdxl) => sdxl.refinerModel);

const optionsFilter = (model: MainModelConfig) => model.base === 'sdxl-refiner';

const ParamSDXLRefinerModelSelect = () => {
  const dispatch = useAppDispatch();
  const model = useAppSelector(selectModel);
  const { t } = useTranslation();
  const { data, isLoading } = useGetMainModelsQuery(REFINER_BASE_MODELS);
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
    modelEntities: data,
    onChange: _onChange,
    selectedModel: model,
    isLoading,
    optionsFilter,
  });
  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length}>
      <InformationalPopover feature="refinerModel">
        <FormLabel>{t('sdxl.refinermodel')}</FormLabel>
      </InformationalPopover>
      <Combobox
        value={value}
        placeholder={placeholder}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
      />
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerModelSelect);
