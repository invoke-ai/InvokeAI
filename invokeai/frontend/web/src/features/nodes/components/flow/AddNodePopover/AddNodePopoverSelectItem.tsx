import { Text } from '@chakra-ui/react';
import { forwardRef } from 'react';
import 'reactflow/dist/style.css';

interface ItemProps extends React.ComponentPropsWithoutRef<'div'> {
  value: string;
  label: string;
  description: string;
}

export const AddNodePopoverSelectItem = forwardRef<HTMLDivElement, ItemProps>(
  ({ label, description, ...others }: ItemProps, ref) => {
    return (
      <div ref={ref} {...others}>
        <div>
          <Text fontWeight={600}>{label}</Text>
          <Text
            size="xs"
            sx={{ color: 'base.600', _dark: { color: 'base.500' } }}
          >
            {description}
          </Text>
        </div>
      </div>
    );
  }
);

AddNodePopoverSelectItem.displayName = 'AddNodePopoverSelectItem';
