import { Switch } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldBooleanValueChanged } from 'features/nodes/store/nodesSlice';
import type { BooleanFieldInputInstance, BooleanFieldInputTemplate } from 'features/nodes/types/field';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

import type { FieldComponentProps } from './types';

const BooleanFieldInputComponent = (
  props: FieldComponentProps<BooleanFieldInputInstance, BooleanFieldInputTemplate>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        fieldBooleanValueChanged({
          nodeId,
          fieldName: field.name,
          value: e.target.checked,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return <Switch className="nodrag" onChange={handleValueChanged} isChecked={field.value}></Switch>;
};

export default memo(BooleanFieldInputComponent);
