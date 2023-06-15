import {
  ClipInputFieldTemplate,
  ClipInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FieldComponentProps } from './types';

const ClipInputFieldComponent = (
  props: FieldComponentProps<ClipInputFieldValue, ClipInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  return null;
};

export default memo(ClipInputFieldComponent);
