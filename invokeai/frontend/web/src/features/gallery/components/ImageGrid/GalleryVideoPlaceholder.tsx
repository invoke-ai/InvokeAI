import { type FlexProps, Flex, Icon } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { PiVideoBold } from 'react-icons/pi';

export const GalleryVideoPlaceholder = memo((props: FlexProps) => (
  <Flex w="full" h="full" bg="base.850" borderRadius="base" alignItems="center" justifyContent="center" {...props}>
    <Icon as={PiVideoBold} boxSize={16} color="base.800" />
  </Flex>
));

GalleryVideoPlaceholder.displayName = 'GalleryVideoPlaceholder';
