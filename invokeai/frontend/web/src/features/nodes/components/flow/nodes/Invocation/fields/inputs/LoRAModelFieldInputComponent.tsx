import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useGroupedModelInvSelect } from 'common/components/InvSelect/useGroupedModelInvSelect';
import { fieldLoRAModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  LoRAModelFieldInputInstance,
  LoRAModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import type { LoRAModelConfigEntity } from 'services/api/endpoints/models';
import { useGetLoRAModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<
  LoRAModelFieldInputInstance,
  LoRAModelFieldInputTemplate
>;

const LoRAModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetLoRAModelsQuery();
  const _onChange = useCallback(
    (value: LoRAModelConfigEntity | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldLoRAModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const { options, value, onChange, placeholder, noOptionsMessage } =
    useGroupedModelInvSelect({
      modelEntities: data,
      onChange: _onChange,
      selectedModel: field.value
        ? { ...field.value, model_type: 'lora' }
        : undefined,
      isLoading,
    });

  return (
    <InvControl
      className="nowheel nodrag"
      isInvalid={!value}
      isDisabled={!options.length}
    >
      <InvSelect
        value={value}
        placeholder={placeholder}
        noOptionsMessage={noOptionsMessage}
        options={options}
        onChange={onChange}
      />
    </InvControl>
  );
};

export default memo(LoRAModelFieldInputComponent);
