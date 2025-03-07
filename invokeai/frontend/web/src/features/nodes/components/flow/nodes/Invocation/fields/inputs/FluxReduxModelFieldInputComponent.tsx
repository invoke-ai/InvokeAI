import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldFluxReduxModelValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { FluxReduxModelFieldInputInstance, FluxReduxModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useFluxReduxModels } from 'services/api/hooks/modelsByType';
import type { FLUXReduxModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const FluxReduxModelFieldInputComponent = (
  props: FieldComponentProps<FluxReduxModelFieldInputInstance, FluxReduxModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useFluxReduxModels();

  const _onChange = useCallback(
    (value: FLUXReduxModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldFluxReduxModelValueChanged({
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
      <FormControl className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`} isInvalid={!value}>
        <Combobox value={value} placeholder="Pick one" options={options} onChange={onChange} />
      </FormControl>
    </Tooltip>
  );
};

export default memo(FluxReduxModelFieldInputComponent);
