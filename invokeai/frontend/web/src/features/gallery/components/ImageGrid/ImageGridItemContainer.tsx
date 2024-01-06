import type { FlexProps } from '@chakra-ui/react';
import { Box, forwardRef } from '@chakra-ui/react';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

// This is exported so that we can use it to calculate the number of images per row
// for the directional gallery navigation.
export const GALLERY_IMAGE_PADDING_PX = 6;

type ItemContainerProps = PropsWithChildren & FlexProps;
const ItemContainer = forwardRef((props: ItemContainerProps, ref) => (
  <Box
    className="item-container"
    ref={ref}
    p={`${GALLERY_IMAGE_PADDING_PX}px`}
    data-testid="image-item-container"
  >
    {props.children}
  </Box>
));

export default memo(ItemContainer);
