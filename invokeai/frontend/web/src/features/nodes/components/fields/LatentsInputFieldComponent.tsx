import { LatentsInputField } from 'features/nodes/types';
import { TbBrandMatrix } from 'react-icons/tb';
import { FieldComponentProps } from './types';

export const LatentsInputFieldComponent = (
  props: FieldComponentProps<LatentsInputField>
) => {
  const { nodeId, field } = props;

  return <TbBrandMatrix />;
};
