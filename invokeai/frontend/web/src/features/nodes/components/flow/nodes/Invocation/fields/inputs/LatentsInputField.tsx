import {
  LatentsInputFieldTemplate,
  LatentsInputFieldValue,
  FieldComponentProps,
  LatentsPolymorphicInputFieldValue,
  LatentsPolymorphicInputFieldTemplate,
} from 'features/nodes/types/types';
import { memo } from 'react';

const LatentsInputFieldComponent = (
  _props: FieldComponentProps<
    LatentsInputFieldValue | LatentsPolymorphicInputFieldValue,
    LatentsInputFieldTemplate | LatentsPolymorphicInputFieldTemplate
  >
) => {
  return null;
};

export default memo(LatentsInputFieldComponent);
