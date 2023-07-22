import {
  ArrayInputFieldTemplate,
  ArrayInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FaList } from 'react-icons/fa';
import { FieldComponentProps } from './types';

const ArrayInputFieldComponent = (
  _props: FieldComponentProps<ArrayInputFieldValue, ArrayInputFieldTemplate>
) => {
  return <FaList />;
};

export default memo(ArrayInputFieldComponent);
