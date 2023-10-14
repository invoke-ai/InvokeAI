import {
  T2IAdapterInputFieldTemplate,
  T2IAdapterInputFieldValue,
  T2IAdapterPolymorphicInputFieldTemplate,
  T2IAdapterPolymorphicInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { memo } from 'react';

const T2IAdapterInputFieldComponent = (
  _props: FieldComponentProps<
    T2IAdapterInputFieldValue | T2IAdapterPolymorphicInputFieldValue,
    T2IAdapterInputFieldTemplate | T2IAdapterPolymorphicInputFieldTemplate
  >
) => {
  return null;
};

export default memo(T2IAdapterInputFieldComponent);
