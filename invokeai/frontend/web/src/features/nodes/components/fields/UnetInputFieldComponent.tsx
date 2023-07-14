import {
  UNetInputFieldTemplate,
  UNetInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FieldComponentProps } from './types';

const UNetInputFieldComponent = (
  props: FieldComponentProps<UNetInputFieldValue, UNetInputFieldTemplate>
) => {
  const { nodeId, field } = props;
  return null;
};

export default memo(UNetInputFieldComponent);
