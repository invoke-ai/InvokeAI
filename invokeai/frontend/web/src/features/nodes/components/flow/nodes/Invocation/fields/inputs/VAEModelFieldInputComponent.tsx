import { Flex } from '@chakra-ui/layout';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useGroupedModelInvSelect } from 'common/components/InvSelect/useGroupedModelInvSelect';
import { SyncModelsIconButton } from 'features/modelManager/components/SyncModels/SyncModelsIconButton';
import { fieldVaeModelValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  VAEModelFieldInputInstance,
  VAEModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback } from 'react';
import type { VaeModelConfigEntity } from 'services/api/endpoints/models';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';

import type { FieldComponentProps } from './types';

type Props = FieldComponentProps<
  VAEModelFieldInputInstance,
  VAEModelFieldInputTemplate
>;

const VAEModelFieldInputComponent = (props: Props) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetVaeModelsQuery();
  const _onChange = useCallback(
    (value: VaeModelConfigEntity | null) => {
      if (!value) {
        return;
      }
      dispatch(
        fieldVaeModelValueChanged({
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
      selectedModel: field.value ? { ...field.value, model_type: 'vae' } : null,
      isLoading,
    });

  return (
    <Flex sx={{ w: 'full', alignItems: 'center', gap: 2 }}>
      <InvControl
        className="nowheel nodrag"
        isDisabled={!options.length}
        isInvalid={!value}
      >
        <InvSelect
          value={value}
          placeholder={placeholder}
          options={options}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </InvControl>
      <SyncModelsIconButton className="nodrag" />
    </Flex>
  );
};

export default memo(VAEModelFieldInputComponent);
