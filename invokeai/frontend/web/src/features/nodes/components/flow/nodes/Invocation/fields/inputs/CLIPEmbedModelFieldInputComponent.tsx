import { Combobox, Flex, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldCLIPEmbedValueChanged } from 'features/nodes/store/nodesSlice';
import type { CLIPEmbedModelFieldInputInstance, CLIPEmbedModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useCLIPEmbedModels } from 'services/api/hooks/modelsByType';
import type { CLIPEmbedModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<CLIPEmbedModelFieldInputInstance, CLIPEmbedModelFieldInputTemplate>;

const CLIPEmbedModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const { t } = useTranslation();
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useCLIPEmbedModels();
  const _onChange = useCallback(
    (value: CLIPEmbedModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldCLIPEmbedValueChanged({
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
      <Tooltip label={!disabledTabs.includes('models') && t('modelManager.starterModelsInModelManager')}>
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

export default memo(CLIPEmbedModelFieldInputComponent);
