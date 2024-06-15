import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { vaeSelected } from 'features/controlLayers/store/canvasV2Slice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useVAEModels } from 'services/api/hooks/modelsByType';
import type { VAEModelConfig } from 'services/api/types';

const ParamVAEModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const model = useAppSelector((s) => s.canvasV2.params.model);
  const vae = useAppSelector((s) => s.canvasV2.params.vae);
  const [modelConfigs, { isLoading }] = useVAEModels();
  const getIsDisabled = useCallback(
    (vae: VAEModelConfig): boolean => {
      const isCompatible = model?.base === vae.base;
      const hasMainModel = Boolean(model?.base);
      return !hasMainModel || !isCompatible;
    },
    [model?.base]
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
