import { Combobox, Flex, FormControl, FormLabel, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import { refinerModelChanged, selectRefinerModel } from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useRefinerModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig } from 'services/api/types';

const optionsFilter = (model: MainModelConfig) => model.base === 'sdxl-refiner';

const ParamSDXLRefinerModelSelect = () => {
  const dispatch = useAppDispatch();
  const model = useAppSelector(selectRefinerModel);
  const { t } = useTranslation();
  const [modelConfigs, { isLoading }] = useRefinerModels();
  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (!model) {
        dispatch(refinerModelChanged(null));
        return;
      }
      dispatch(refinerModelChanged(zModelIdentifierField.parse(model)));
    },
    [dispatch]
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
    <FormControl isDisabled={!options.length} isInvalid={!options.length} w="full">
      <InformationalPopover feature="refinerModel">
        <FormLabel>{t('sdxl.refinermodel')}</FormLabel>
      </InformationalPopover>
      <Flex w="full" minW={0} gap={2} alignItems="center">
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
      </Flex>
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerModelSelect);
