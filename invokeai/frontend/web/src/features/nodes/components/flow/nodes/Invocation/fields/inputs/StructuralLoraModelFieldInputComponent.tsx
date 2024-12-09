import { Combobox, Flex, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldStructuralLoRAModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  StructuralLoRAModelFieldInputInstance,
  StructuralLoRAModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useStructuralLoRAModel } from 'services/api/hooks/modelsByType';
import { isStructuralLoRAModelConfig, type StructuralLoRAModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<StructuralLoRAModelFieldInputInstance, StructuralLoRAModelFieldInputTemplate>;

const StructuralLoRAModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const { t } = useTranslation();
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useStructuralLoRAModel();

  const _onChange = useCallback(
    (value: StructuralLoRAModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldStructuralLoRAModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs: modelConfigs.filter((config) => isStructuralLoRAModelConfig(config)),
    onChange: _onChange,
    isLoading,
    selectedModel: field.value,
  });
  const required = props.fieldTemplate.required;

  return (
    <Flex w="full" alignItems="center" gap={2}>
      <Tooltip label={!disabledTabs.includes('models') && t('modelManager.starterModelsInModelManager')}>
        <FormControl className="nowheel nodrag" isDisabled={!options.length} isInvalid={!value && required}>
          <Combobox
            value={value}
            placeholder={required ? placeholder : `(Optional) ${placeholder}`}
            options={options}
            onChange={onChange}
            noOptionsMessage={noOptionsMessage}
          />
        </FormControl>
      </Tooltip>
    </Flex>
  );
};

export default memo(StructuralLoRAModelFieldInputComponent);
