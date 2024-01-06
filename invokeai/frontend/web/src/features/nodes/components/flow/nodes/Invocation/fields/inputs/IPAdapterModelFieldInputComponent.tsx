import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useGroupedModelInvSelect } from 'common/components/InvSelect/useGroupedModelInvSelect';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { fieldIPAdapterModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  IPAdapterModelFieldInputInstance,
  IPAdapterModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import type { IPAdapterModelConfigEntity } from 'services/api/endpoints/models';
import { useGetIPAdapterModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

const IPAdapterModelFieldInputComponent = (
  props: FieldComponentProps<
    IPAdapterModelFieldInputInstance,
    IPAdapterModelFieldInputTemplate
  >
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data: ipAdapterModels } = useGetIPAdapterModelsQuery();

  const _onChange = useCallback(
    (value: IPAdapterModelConfigEntity | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldIPAdapterModelValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const { options, value, onChange } = useGroupedModelInvSelect({
    modelEntities: ipAdapterModels,
    onChange: _onChange,
    selectedModel: field.value
      ? { ...field.value, model_type: 'ip_adapter' }
      : undefined,
  });

  return (
    <InvTooltip label={value?.description}>
      <InvControl className="nowheel nodrag" isInvalid={!value}>
        <InvSelect
          value={value}
          placeholder="Pick one"
          options={options}
          onChange={onChange}
        />
      </InvControl>
    </InvTooltip>
  );
};

export default memo(IPAdapterModelFieldInputComponent);
