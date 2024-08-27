import { Combobox, Flex, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldT5EncoderValueChanged } from 'features/nodes/store/nodesSlice';
import type { T5EncoderModelFieldInputInstance, T5EncoderModelFieldInputTemplate } from 'features/nodes/types/field';
import { selectIsModelsTabDisabled } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useT5EncoderModels } from 'services/api/hooks/modelsByType';
import type { T5EncoderBnbQuantizedLlmInt8bModelConfig, T5EncoderModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<T5EncoderModelFieldInputInstance, T5EncoderModelFieldInputTemplate>;

const T5EncoderModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const { t } = useTranslation();
  const isModelsTabDisabled = useAppSelector(selectIsModelsTabDisabled);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useT5EncoderModels();
  const _onChange = useCallback(
    (value: T5EncoderBnbQuantizedLlmInt8bModelConfig | T5EncoderModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldT5EncoderValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    isLoading,
    selectedModel: field.value,
  });

  return (
    <Flex w="full" alignItems="center" gap={2}>
      <Tooltip label={!isModelsTabDisabled && t('modelManager.starterModelsInModelManager')}>
        <FormControl className="nowheel nodrag" isDisabled={!options.length} isInvalid={!value}>
          <Combobox
            value={value}
            placeholder={placeholder}
            options={options}
            onChange={onChange}
            noOptionsMessage={noOptionsMessage}
          />
        </FormControl>
      </Tooltip>
    </Flex>
  );
};

export default memo(T5EncoderModelFieldInputComponent);
