import { Box, FlexProps, forwardRef } from '@chakra-ui/react';
import { PropsWithChildren, memo } from 'react';

type ItemContainerProps = PropsWithChildren & FlexProps;
const ItemContainer = forwardRef((props: ItemContainerProps, ref) => (
  <Box className="item-container" ref={ref} p={1.5}>
    {props.children}
  </Box>
));

export default memo(ItemContainer);
