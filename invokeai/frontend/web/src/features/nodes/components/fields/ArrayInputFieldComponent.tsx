import {
  ArrayInputFieldTemplate,
  ArrayInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FaList } from 'react-icons/fa';
import { FieldComponentProps } from './types';

const ArrayInputFieldComponent = (
  props: FieldComponentProps<ArrayInputFieldValue, ArrayInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  return <FaList />;
};

export default memo(ArrayInputFieldComponent);
