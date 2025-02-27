import { Combobox, Flex, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldControlLoRAModelValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type {
  ControlLoRAModelFieldInputInstance,
  ControlLoRAModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useControlLoRAModel } from 'services/api/hooks/modelsByType';
import { type ControlLoRAModelConfig, isControlLoRAModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<ControlLoRAModelFieldInputInstance, ControlLoRAModelFieldInputTemplate>;

const ControlLoRAModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const { t } = useTranslation();
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useControlLoRAModel();

  const _onChange = useCallback(
    (value: ControlLoRAModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldControlLoRAModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs: modelConfigs.filter((config) => isControlLoRAModelConfig(config)),
    onChange: _onChange,
    isLoading,
    selectedModel: field.value,
  });
  const required = props.fieldTemplate.required;

  return (
    <Flex w="full" alignItems="center" gap={2}>
      <Tooltip label={!disabledTabs.includes('models') && t('modelManager.starterModelsInModelManager')}>
        <FormControl
          className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
          isDisabled={!options.length}
          isInvalid={!value && required}
        >
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

export default memo(ControlLoRAModelFieldInputComponent);
