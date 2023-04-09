import { Switch } from '@chakra-ui/react';
import { useAppDispatch } from 'app/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { BooleanInputField } from 'features/nodes/types';
import { ChangeEvent } from 'react';
import { FieldComponentProps } from './types';

export const BooleanInputFieldComponent = (
  props: FieldComponentProps<BooleanInputField>
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const handleValueChanged = (e: ChangeEvent<HTMLInputElement>) => {
    dispatch(
      fieldValueChanged({
        nodeId,
        fieldId: field.name,
        value: e.target.checked,
      })
    );
  };

  return (
    <Switch onChange={handleValueChanged} isChecked={field.value}></Switch>
  );
};
