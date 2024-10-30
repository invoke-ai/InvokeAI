import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fluxVAESelected, selectFLUXVAE } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useFluxVAEModels } from 'services/api/hooks/modelsByType';
import type { VAEModelConfig } from 'services/api/types';

const ParamFLUXVAEModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const vae = useAppSelector(selectFLUXVAE);
  const [modelConfigs, { isLoading }] = useFluxVAEModels();

  const _onChange = useCallback(
    (vae: VAEModelConfig | null) => {
      if (vae) {
        dispatch(fluxVAESelected(zModelIdentifierField.parse(vae)));
      }
    },
    [dispatch]
  );

  const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: vae,
    isLoading,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} minW={0} flexGrow={1}>
      <InformationalPopover feature="paramVAE">
        <FormLabel m={0}>{t('modelManager.vae')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} noOptionsMessage={noOptionsMessage} />
    </FormControl>
  );
};

export default memo(ParamFLUXVAEModelSelect);
