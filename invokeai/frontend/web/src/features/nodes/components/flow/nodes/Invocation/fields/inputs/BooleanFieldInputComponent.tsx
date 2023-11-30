import { Switch } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldBooleanValueChanged } from 'features/nodes/store/nodesSlice';
import {
  BooleanFieldInputInstance,
  BooleanFieldInputTemplate,
} from 'features/nodes/types/field';
import { FieldComponentProps } from './types';
import { ChangeEvent, memo, useCallback } from 'react';

const BooleanFieldInputComponent = (
  props: FieldComponentProps<
    BooleanFieldInputInstance,
    BooleanFieldInputTemplate
  >
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

  return (
    <Switch
      className="nodrag"
      onChange={handleValueChanged}
      isChecked={field.value}
    ></Switch>
  );
};

export default memo(BooleanFieldInputComponent);
