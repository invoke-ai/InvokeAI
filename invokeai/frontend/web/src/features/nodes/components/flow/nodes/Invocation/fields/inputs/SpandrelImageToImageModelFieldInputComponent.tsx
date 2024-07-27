import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldSpandrelImageToImageModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  SpandrelImageToImageModelFieldInputInstance,
  SpandrelImageToImageModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useSpandrelImageToImageModels } from 'services/api/hooks/modelsByType';
import type { SpandrelImageToImageModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const SpandrelImageToImageModelFieldInputComponent = (
  props: FieldComponentProps<SpandrelImageToImageModelFieldInputInstance, SpandrelImageToImageModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useSpandrelImageToImageModels();

  const _onChange = useCallback(
    (value: SpandrelImageToImageModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldSpandrelImageToImageModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const { options, value, onChange } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: field.value,
    isLoading,
  });

  return (
    <Tooltip label={value?.description}>
      <FormControl className="nowheel nodrag" isInvalid={!value}>
        <Combobox value={value} placeholder="Pick one" options={options} onChange={onChange} />
      </FormControl>
    </Tooltip>
  );
};

export default memo(SpandrelImageToImageModelFieldInputComponent);
