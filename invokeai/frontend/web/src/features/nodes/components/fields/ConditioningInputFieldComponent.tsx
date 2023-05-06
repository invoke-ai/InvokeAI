import {
  ConditioningInputFieldTemplate,
  ConditioningInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FieldComponentProps } from './types';

const ConditioningInputFieldComponent = (
  props: FieldComponentProps<
    ConditioningInputFieldValue,
    ConditioningInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;

  return null;
};

export default memo(ConditioningInputFieldComponent);
