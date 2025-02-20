import { Combobox, Flex, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldCLIPGEmbedValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { CLIPGEmbedModelFieldInputInstance, CLIPGEmbedModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCLIPEmbedModels } from 'services/api/hooks/modelsByType';
import { type CLIPGEmbedModelConfig, isCLIPGEmbedModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<CLIPGEmbedModelFieldInputInstance, CLIPGEmbedModelFieldInputTemplate>;

const CLIPGEmbedModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const { t } = useTranslation();
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useCLIPEmbedModels();

  const _onChange = useCallback(
    (value: CLIPGEmbedModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldCLIPGEmbedValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );
  const { options, value, onChange, placeholder, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs: modelConfigs.filter((config) => isCLIPGEmbedModelConfig(config)),
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

export default memo(CLIPGEmbedModelFieldInputComponent);
