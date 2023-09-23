import { Box, Text } from '@chakra-ui/react';
import { forwardRef, memo } from 'react';

interface ItemProps extends React.ComponentPropsWithoutRef<'div'> {
  label: string;
  value: string;
  description?: string;
}

const IAIMantineSelectItemWithDescription = forwardRef<
  HTMLDivElement,
  ItemProps
>(({ label, description, ...rest }: ItemProps, ref) => (
  <Box ref={ref} {...rest}>
    <Box>
      <Text fontWeight={600}>{label}</Text>
      {description && (
        <Text size="xs" variant="subtext">
          {description}
        </Text>
      )}
    </Box>
  </Box>
));

IAIMantineSelectItemWithDescription.displayName =
  'IAIMantineSelectItemWithDescription';

export default memo(IAIMantineSelectItemWithDescription);
