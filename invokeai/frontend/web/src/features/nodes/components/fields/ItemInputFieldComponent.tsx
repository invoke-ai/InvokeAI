import {
  ItemInputFieldTemplate,
  ItemInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FaAddressCard } from 'react-icons/fa';
import { FieldComponentProps } from './types';

const ItemInputFieldComponent = (
  props: FieldComponentProps<ItemInputFieldValue, ItemInputFieldTemplate>
) => {
  const { nodeId, field } = props;

  return <FaAddressCard />;
};

export default memo(ItemInputFieldComponent);
