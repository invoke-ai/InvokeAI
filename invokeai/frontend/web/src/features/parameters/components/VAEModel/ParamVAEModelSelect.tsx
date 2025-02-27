import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectBase, selectVAE, vaeSelected } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useVAEModels } from 'services/api/hooks/modelsByType';
import type { VAEModelConfig } from 'services/api/types';

const ParamVAEModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const base = useAppSelector(selectBase);
  const vae = useAppSelector(selectVAE);
  const [modelConfigs, { isLoading }] = useVAEModels();
  const getIsDisabled = useCallback(
    (vae: VAEModelConfig): boolean => {
      const isCompatible = base === vae.base;
      const hasMainModel = Boolean(base);
      return !hasMainModel || !isCompatible;
    },
    [base]
  );
  const _onChange = useCallback(
    (vae: VAEModelConfig | null) => {
      dispatch(vaeSelected(vae ? zModelIdentifierField.parse(vae) : null));
    },
    [dispatch]
  );
  const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: vae,
    isLoading,
    getIsDisabled,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} minW={0} flexGrow={1}>
      <InformationalPopover feature="paramVAE">
        <FormLabel m={0}>{t('modelManager.vae')}</FormLabel>
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
