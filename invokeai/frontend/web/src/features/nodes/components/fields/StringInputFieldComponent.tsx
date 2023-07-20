import { Textarea } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  StringInputFieldTemplate,
  StringInputFieldValue,
} from 'features/nodes/types/types';
import { ChangeEvent, memo } from 'react';
import { FieldComponentProps } from './types';

const StringInputFieldComponent = (
  props: FieldComponentProps<
    StringInputFieldValue,
    StringInputFieldTemplate
  > & {
    nodeWidth: number;
  }
) => {
  const { nodeId, field, nodeWidth } = props;
  const dispatch = useAppDispatch();

  const handleValueChanged = (e: ChangeEvent<HTMLTextAreaElement>) => {
    dispatch(
      fieldValueChanged({
        nodeId,
        fieldName: field.name,
        value: e.target.value,
      })
    );
  };

  const textareaWidth = nodeWidth - 20;

  return (
    <Textarea
      style={{
        height: '150px',
        width: `${textareaWidth}px`,
        resize: 'none',
        overflowY: 'auto',
      }}
      onChange={handleValueChanged}
      value={field.value}
    />
  );
};

export default memo(StringInputFieldComponent);
