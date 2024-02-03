import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { refinerModelChanged, selectSdxlSlice } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { REFINER_BASE_MODELS } from 'services/api/constants';
import type { MainModelConfigEntity } from 'services/api/endpoints/models';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

const selectModel = createMemoizedSelector(selectSdxlSlice, (sdxl) => sdxl.refinerModel);

const optionsFilter = (model: MainModelConfigEntity) => model.base_model === 'sdxl-refiner';

const ParamSDXLRefinerModelSelect = () => {
  const dispatch = useAppDispatch();
  const model = useAppSelector(selectModel);
  const { t } = useTranslation();
  const { data, isLoading } = useGetMainModelsQuery(REFINER_BASE_MODELS);
  const _onChange = useCallback(
    (model: MainModelConfigEntity | null) => {
      if (!model) {
        dispatch(refinerModelChanged(null));
        return;
      }
      dispatch(
        refinerModelChanged({
          base_model: 'sdxl-refiner',
          model_name: model.model_name,
          model_type: model.model_type,
        })
      );
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
      <FormLabel>{t('sdxl.refinermodel')}</FormLabel>
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
