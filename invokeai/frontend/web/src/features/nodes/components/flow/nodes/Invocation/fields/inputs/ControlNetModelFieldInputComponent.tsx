import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useGroupedModelInvSelect } from 'common/components/InvSelect/useGroupedModelInvSelect';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { fieldControlNetModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  ControlNetModelFieldInputInstance,
  ControlNetModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import type { ControlNetModelConfigEntity } from 'services/api/endpoints/models';
import { useGetControlNetModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<
  ControlNetModelFieldInputInstance,
  ControlNetModelFieldInputTemplate
>;

const ControlNetModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetControlNetModelsQuery();

  const _onChange = useCallback(
    (value: ControlNetModelConfigEntity | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldControlNetModelValueChanged({
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
        ? { ...field.value, model_type: 'controlnet' }
        : undefined,
      isLoading,
    });

  return (
    <InvTooltip label={value?.description}>
      <InvControl className="nowheel nodrag" isInvalid={!value}>
        <InvSelect
          value={value}
          placeholder={placeholder}
          options={options}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </InvControl>
    </InvTooltip>
  );
};

export default memo(ControlNetModelFieldInputComponent);
