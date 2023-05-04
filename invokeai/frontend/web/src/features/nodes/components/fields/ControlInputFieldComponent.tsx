import {
  ControlInputFieldTemplate,
  ControlInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FieldComponentProps } from './types';

const ControlInputFieldComponent = (
  props: FieldComponentProps<ControlInputFieldValue, ControlInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  return null;
};

export default memo(ControlInputFieldComponent);
