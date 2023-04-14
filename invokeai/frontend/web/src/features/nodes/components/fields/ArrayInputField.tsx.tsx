import { ArrayInputField, ImageInputField } from 'features/nodes/types';
import { FaImage, FaList } from 'react-icons/fa';
import { FieldComponentProps } from './types';

export const ArrayInputFieldComponent = (
  props: FieldComponentProps<ArrayInputField>
) => {
  const { nodeId, field } = props;

  return <FaList />;
};
