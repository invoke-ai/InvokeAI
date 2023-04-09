import { ImageInputField } from 'features/nodes/types';
import { FaImage } from 'react-icons/fa';
import { FieldComponentProps } from './types';

export const ImageInputFieldComponent = (
  props: FieldComponentProps<ImageInputField>
) => {
  const { nodeId, field } = props;

  return <FaImage />;
};
