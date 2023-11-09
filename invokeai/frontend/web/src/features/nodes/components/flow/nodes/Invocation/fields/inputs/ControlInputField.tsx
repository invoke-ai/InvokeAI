import {
  ControlInputFieldTemplate,
  ControlInputFieldValue,
  ControlPolymorphicInputFieldTemplate,
  ControlPolymorphicInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { memo } from 'react';

const ControlInputFieldComponent = (
  _props: FieldComponentProps<
    ControlInputFieldValue | ControlPolymorphicInputFieldValue,
    ControlInputFieldTemplate | ControlPolymorphicInputFieldTemplate
  >
) => {
  return null;
};

export default memo(ControlInputFieldComponent);
