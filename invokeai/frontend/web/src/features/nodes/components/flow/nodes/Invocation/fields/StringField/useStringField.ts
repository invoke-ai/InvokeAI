import { useAppDispatch } from 'app/store/storeHooks';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { fieldStringValueChanged } from 'features/nodes/store/nodesSlice';
import type { StringFieldInputInstance, StringFieldInputTemplate } from 'features/nodes/types/field';
import type { ChangeEvent } from 'react';
import { useCallback } from 'react';

export const useStringField = (props: FieldComponentProps<StringFieldInputInstance, StringFieldInputTemplate>) => {
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      dispatch(
        fieldStringValueChanged({
          nodeId,
          fieldName: field.name,
          value: e.target.value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return {
    value: field.value,
    onChange,
    defaultValue: fieldTemplate.default,
  };
};
