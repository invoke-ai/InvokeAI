import type { FlexProps } from '@chakra-ui/react';
import { Box, forwardRef } from '@chakra-ui/react';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type ItemContainerProps = PropsWithChildren & FlexProps;
const ItemContainer = forwardRef((props: ItemContainerProps, ref) => (
  <Box
    className="item-container"
    ref={ref}
    p={1.5}
    data-testid="image-item-container"
  >
    {props.children}
  </Box>
));

export default memo(ItemContainer);
