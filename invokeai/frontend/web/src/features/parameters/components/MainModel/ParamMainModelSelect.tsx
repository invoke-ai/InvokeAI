import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useGroupedModelInvSelect } from 'common/components/InvSelect/useGroupedModelInvSelect';
import { modelSelected } from 'features/parameters/store/actions';
import { selectGenerationSlice } from 'features/parameters/store/generationSlice';
import { pick } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { NON_REFINER_BASE_MODELS } from 'services/api/constants';
import type { MainModelConfigEntity } from 'services/api/endpoints/models';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

const selectModel = createMemoizedSelector(
  selectGenerationSlice,
  (generation) => generation.model
);

const ParamMainModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const model = useAppSelector(selectModel);
  const { data, isLoading } = useGetMainModelsQuery(NON_REFINER_BASE_MODELS);
  const _onChange = useCallback(
    (model: MainModelConfigEntity | null) => {
      if (!model) {
        return;
      }
      dispatch(
        modelSelected(pick(model, ['base_model', 'model_name', 'model_type']))
      );
    },
    [dispatch]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } =
    useGroupedModelInvSelect({
      modelEntities: data,
      onChange: _onChange,
      selectedModel: model,
      isLoading,
    });

  return (
    <InvControl
      label={t('modelManager.model')}
      isDisabled={!options.length}
      isInvalid={!options.length}
    >
      <InvSelect
        value={value}
        placeholder={placeholder}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
      />
    </InvControl>
  );
};

export default memo(ParamMainModelSelect);
