import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectGenerationSlice, vaeSelected } from 'features/parameters/store/generationSlice';
import { pick } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { VaeModelConfigEntity } from 'services/api/endpoints/models';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';

const selector = createMemoizedSelector(selectGenerationSlice, (generation) => {
  const { model, vae } = generation;
  return { model, vae };
});

const ParamVAEModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { model, vae } = useAppSelector(selector);
  const { data, isLoading } = useGetVaeModelsQuery();
  const getIsDisabled = useCallback(
    (vae: VaeModelConfigEntity): boolean => {
      const isCompatible = model?.base_model === vae.base_model;
      const hasMainModel = Boolean(model?.base_model);
      return !hasMainModel || !isCompatible;
    },
    [model?.base_model]
  );
  const _onChange = useCallback(
    (vae: VaeModelConfigEntity | null) => {
      dispatch(vaeSelected(vae ? pick(vae, 'base_model', 'model_name') : null));
    },
    [dispatch]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelEntities: data,
    onChange: _onChange,
    selectedModel: vae ? { ...vae, model_type: 'vae' } : null,
    isLoading,
    getIsDisabled,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length}>
      <InformationalPopover feature="paramVAE">
        <FormLabel>{t('modelManager.vae')}</FormLabel>
      </InformationalPopover>
      <Combobox
        isClearable
        value={value}
        placeholder={value ? placeholder : t('models.defaultVAE')}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
      />
    </FormControl>
  );
};

export default memo(ParamVAEModelSelect);
