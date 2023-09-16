import { useAppDispatch } from 'app/store/storeHooks';
import IAIInput from 'common/components/IAIInput';
import IAITextarea from 'common/components/IAITextarea';
import { fieldStringValueChanged } from 'features/nodes/store/nodesSlice';
import {
  StringInputFieldTemplate,
  StringInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { ChangeEvent, memo, useCallback } from 'react';

const StringInputFieldComponent = (
  props: FieldComponentProps<StringInputFieldValue, StringInputFieldTemplate>
) => {
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();

  const handleValueChanged = useCallback(
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

  if (fieldTemplate.ui_component === 'textarea') {
    return (
      <IAITextarea
        className="nodrag"
        onChange={handleValueChanged}
        value={field.value}
        rows={5}
        resize="none"
      />
    );
  }

  return <IAIInput onChange={handleValueChanged} value={field.value} />;
};

export default memo(StringInputFieldComponent);
