import { Input } from '@chakra-ui/react';
import { useAppDispatch } from 'app/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { StringInputField } from 'features/nodes/types';
import { ChangeEvent } from 'react';
import { FieldComponentProps } from './types';

export const StringInputFieldComponent = (
  props: FieldComponentProps<StringInputField>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = (e: ChangeEvent<HTMLInputElement>) => {
    dispatch(
      fieldValueChanged({ nodeId, fieldId: field.name, value: e.target.value })
    );
  };

  return <Input onChange={handleValueChanged} value={field.value}></Input>;
};
