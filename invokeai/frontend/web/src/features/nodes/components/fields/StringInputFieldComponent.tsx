import { Input, Textarea } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  StringInputFieldTemplate,
  StringInputFieldValue,
} from 'features/nodes/types/types';
import { ChangeEvent, memo } from 'react';
import { FieldComponentProps } from './types';

const StringInputFieldComponent = (
  props: FieldComponentProps<StringInputFieldValue, StringInputFieldTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const handleValueChanged = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    dispatch(
      fieldValueChanged({
        nodeId,
        fieldName: field.name,
        value: e.target.value,
      })
    );
  };

  return ['prompt', 'style'].includes(field.name.toLowerCase()) ? (
    <Textarea onChange={handleValueChanged} value={field.value} rows={2} />
  ) : (
    <Input onChange={handleValueChanged} value={field.value} />
  );
};

export default memo(StringInputFieldComponent);
