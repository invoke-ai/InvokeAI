import {
  ItemInputFieldTemplate,
  ItemInputFieldValue,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { FaAddressCard } from 'react-icons/fa';
import { FieldComponentProps } from './types';

const ItemInputFieldComponent = (
  _props: FieldComponentProps<ItemInputFieldValue, ItemInputFieldTemplate>
) => {
  return <FaAddressCard />;
};

export default memo(ItemInputFieldComponent);
