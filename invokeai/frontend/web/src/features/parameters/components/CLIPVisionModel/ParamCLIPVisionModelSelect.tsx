import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { useControlAdapterCLIPVisionModel } from 'features/controlAdapters/hooks/useControlAdapterCLIPVisionModel';
import { controlAdapterCLIPVisionModelChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCLIPVisionModels } from 'services/api/hooks/modelsByType';
import type { CLIPVisionModelConfig } from 'services/api/types';

type ParamCLIPVisionModelProps = {
  id: string;
};

const ParamCLIPVisionModelSelect = (props: ParamCLIPVisionModelProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const currentCLIPVisionModel = useControlAdapterCLIPVisionModel(props.id);
  const [modelConfigs, { isLoading }] = useCLIPVisionModels();

  const _onChange = useCallback(
    (clipVisionModel: CLIPVisionModelConfig | null) => {
      dispatch(
        controlAdapterCLIPVisionModelChanged({
          id: props.id,
          clipVisionModel: clipVisionModel ? zModelIdentifierField.parse(clipVisionModel) : null,
        })
      );
    },
    [dispatch, props.id]
  );

  const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: currentCLIPVisionModel,
    isLoading,
  });

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length}>
      <InformationalPopover feature="paramVAE">
        <FormLabel>{t('modelManager.clipVisionModel')}</FormLabel>
      </InformationalPopover>
      <Combobox
        isClearable
        value={value}
        placeholder={value ? value.value : t('models.selectModel')}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
      />
    </FormControl>
  );
};

export default memo(ParamCLIPVisionModelSelect);
