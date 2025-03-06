import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { fieldSigLipModelValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { SigLipModelFieldInputInstance, SigLipModelFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import { useSigLipModels } from 'services/api/hooks/modelsByType';
import type { SigLipModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const SigLipModelFieldInputComponent = (
  props: FieldComponentProps<SigLipModelFieldInputInstance, SigLipModelFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const [modelConfigs, { isLoading }] = useSigLipModels();

  const _onChange = useCallback(
    (value: SigLipModelConfig | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldSigLipModelValueChanged({
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

export default memo(SigLipModelFieldInputComponent);
