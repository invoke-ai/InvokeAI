import { Select } from '@chakra-ui/react';
import { useAppDispatch } from 'app/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { EnumInputField } from 'features/nodes/types';
import { ChangeEvent } from 'react';
import { FieldComponentProps } from './types';

export const EnumInputFieldComponent = (
  props: FieldComponentProps<EnumInputField>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = (e: ChangeEvent<HTMLSelectElement>) => {
    dispatch(
      fieldValueChanged({
        nodeId,
        fieldId: field.name,
        value: e.target.value,
      })
    );
  };

  return (
    <Select onChange={handleValueChanged} value={field.value}>
      {field.options.map((option) => (
        <option key={option}>{option}</option>
      ))}
    </Select>
  );
};
