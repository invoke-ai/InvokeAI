import { Combobox, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { refinerModelChanged, selectRefinerModel, useParamsDispatch } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useRefinerModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig } from 'services/api/types';

const optionsFilter = (model: MainModelConfig) => model.base === 'sdxl-refiner';

const ParamSDXLRefinerModelSelect = () => {
  const dispatchParams = useParamsDispatch();
  const model = useAppSelector(selectRefinerModel);
  const { t } = useTranslation();
  const [modelConfigs, { isLoading }] = useRefinerModels();
  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (!model) {
        dispatchParams(refinerModelChanged, null);
        return;
      }
      dispatchParams(refinerModelChanged, zModelIdentifierField.parse(model));
    },
    [dispatchParams]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: model,
    isLoading,
    optionsFilter,
  });
  const onReset = useCallback(() => {
    _onChange(null);
  }, [_onChange]);

  return (
    <FormControl isDisabled={!options.length} isInvalid={!options.length} w="full" gap={2}>
      <InformationalPopover feature="refinerModel">
        <FormLabel>{t('sdxl.refinermodel')}</FormLabel>
      </InformationalPopover>
      <Combobox
        value={value}
        placeholder={placeholder}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
      />
      <IconButton
        size="sm"
        variant="ghost"
        icon={<PiXBold />}
        aria-label={t('common.reset')}
        onClick={onReset}
        isDisabled={!value}
      />
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerModelSelect);
