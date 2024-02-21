import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectGenerationSlice, vaeSelected } from 'features/parameters/store/generationSlice';
import { pick } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';
import type { VAEConfig } from 'services/api/types';

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
    (vae: VAEConfig): boolean => {
      const isCompatible = model?.base === vae.base;
      const hasMainModel = Boolean(model?.base);
      return !hasMainModel || !isCompatible;
    },
    [model?.base]
  );
  const _onChange = useCallback(
    (vae: VAEConfig | null) => {
      dispatch(vaeSelected(vae ? pick(vae, 'key', 'base') : null));
    },
    [dispatch]
  );
  const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
    modelEntities: data,
    onChange: _onChange,
    selectedModel: vae ? pick(vae, 'key', 'base') : null,
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
        placeholder={value ? value.value : t('models.defaultVAE')}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
      />
    </FormControl>
  );
};

export default memo(ParamVAEModelSelect);
