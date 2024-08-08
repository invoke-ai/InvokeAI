import { Input, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldStringValueChanged } from 'features/nodes/store/nodesSlice';
import type { StringFieldInputInstance, StringFieldInputTemplate } from 'features/nodes/types/field';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

import type { FieldComponentProps } from './types';

const StringFieldInputComponent = (props: FieldComponentProps<StringFieldInputInstance, StringFieldInputTemplate>) => {
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
    return <Textarea className="nodrag" onChange={handleValueChanged} value={field.value} rows={5} resize="vertical" />;
  }

  return <Input className="nodrag" onChange={handleValueChanged} value={field.value} />;
};

export default memo(StringFieldInputComponent);
