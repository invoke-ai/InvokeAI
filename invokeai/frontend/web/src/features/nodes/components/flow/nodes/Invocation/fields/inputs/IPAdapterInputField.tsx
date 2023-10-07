import {
  IPAdapterInputFieldTemplate,
  IPAdapterInputFieldValue,
  FieldComponentProps,
  IPAdapterPolymorphicInputFieldValue,
  IPAdapterPolymorphicInputFieldTemplate,
} from 'features/nodes/types/types';
import { memo } from 'react';

const IPAdapterInputFieldComponent = (
  _props: FieldComponentProps<
    IPAdapterInputFieldValue | IPAdapterPolymorphicInputFieldValue,
    IPAdapterInputFieldTemplate | IPAdapterPolymorphicInputFieldTemplate
  >
) => {
  return null;
};

export default memo(IPAdapterInputFieldComponent);
