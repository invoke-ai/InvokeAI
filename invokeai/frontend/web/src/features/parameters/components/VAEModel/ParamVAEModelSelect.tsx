import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useGroupedModelInvSelect } from 'common/components/InvSelect/useGroupedModelInvSelect';
import { vaeSelected } from 'features/parameters/store/generationSlice';
import { pick } from 'lodash-es';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { VaeModelConfigEntity } from 'services/api/endpoints/models';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';

const selector = createMemoizedSelector(stateSelector, ({ generation }) => {
  const { model, vae } = generation;
  return { model, vae };
});

const ParamVAEModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { model, vae } = useAppSelector(selector);
  const { data, isLoading } = useGetVaeModelsQuery();
  const getIsDisabled = (vae: VaeModelConfigEntity): boolean => {
    const isCompatible = model?.base_model === vae.base_model;
    const hasMainModel = Boolean(model?.base_model);
    return !hasMainModel || !isCompatible;
  };
  const _onChange = useCallback(
    (vae: VaeModelConfigEntity | null) => {
      dispatch(vaeSelected(vae ? pick(vae, 'base_model', 'model_name') : null));
    },
    [dispatch]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } =
    useGroupedModelInvSelect({
      modelEntities: data,
      onChange: _onChange,
      selectedModel: vae ? { ...vae, model_type: 'vae' } : null,
      isLoading,
      getIsDisabled,
    });

  return (
    <InvControl
      label={t('modelManager.vae')}
      isDisabled={!options.length}
      isInvalid={!options.length}
      feature="paramVAE"
    >
      <InvSelect
        isClearable
        value={value}
        placeholder={value ? placeholder : t('models.defaultVAE')}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
      />
    </InvControl>
  );
};

export default ParamVAEModelSelect;
