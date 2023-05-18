import { Select } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  EnumInputFieldTemplate,
  EnumInputFieldValue,
} from 'features/nodes/types/types';
import { ChangeEvent, memo } from 'react';
import { FieldComponentProps } from './types';

const EnumInputFieldComponent = (
  props: FieldComponentProps<EnumInputFieldValue, EnumInputFieldTemplate>
) => {
  const { nodeId, field, template } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = (e: ChangeEvent<HTMLSelectElement>) => {
    dispatch(
      fieldValueChanged({
        nodeId,
        fieldName: field.name,
        value: e.target.value,
      })
    );
  };

  return (
    <Select onChange={handleValueChanged} value={field.value}>
      {template.options.map((option) => (
        <option key={option}>{option}</option>
      ))}
    </Select>
  );
};

export default memo(EnumInputFieldComponent);
