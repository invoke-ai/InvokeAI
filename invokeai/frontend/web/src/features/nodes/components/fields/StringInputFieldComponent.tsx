import { memo, ChangeEvent } from 'react';
import { Textarea, Input } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  StringInputFieldTemplate,
  StringInputFieldValue,
} from 'features/nodes/types/types';
import { FieldComponentProps } from './types';

const FIELD_PADDING = 20;

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

  const textareaWidth = nodeWidth - FIELD_PADDING;

  const textareaFieldNames = ['prompt', 'text'];

  return (
    <>
      {textareaFieldNames.includes(field.name.toLowerCase()) ? (
        <Textarea
          style={{
            height: '150px',
            width: `${textareaWidth}px`,
            overflowY: 'auto',
          }}
          onChange={handleValueChanged}
          value={field.value}
        />
      ) : (
        <Input
          style={{ width: `${textareaWidth}px` }}
          onChange={handleValueChanged}
          value={field.value}
        />
      )}
    </>
  );
};

export default memo(StringInputFieldComponent);
