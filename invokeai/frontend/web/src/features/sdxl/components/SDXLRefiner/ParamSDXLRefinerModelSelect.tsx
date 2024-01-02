import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useModelInvSelect } from 'common/components/InvSelect/useModelInvSelect';
import { refinerModelChanged } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { REFINER_BASE_MODELS } from 'services/api/constants';
import type { MainModelConfigEntity } from 'services/api/endpoints/models';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

const selector = createMemoizedSelector(stateSelector, (state) => ({
  model: state.sdxl.refinerModel,
}));

const optionsFilter = (model: MainModelConfigEntity) =>
  model.base_model === 'sdxl-refiner';

const ParamSDXLRefinerModelSelect = () => {
  const dispatch = useAppDispatch();
  const { model } = useAppSelector(selector);
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
  const { options, value, onChange, placeholder, noOptionsMessage } =
    useModelInvSelect({
      modelEntities: data,
      onChange: _onChange,
      selectedModel: model,
      isLoading,
      optionsFilter,
    });
  return (
    <InvControl
      label={t('sdxl.refinermodel')}
      isDisabled={!options.length}
      isInvalid={!options.length}
    >
      <InvSelect
        value={value}
        placeholder={placeholder}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
      />
    </InvControl>
  );
};

export default memo(ParamSDXLRefinerModelSelect);
