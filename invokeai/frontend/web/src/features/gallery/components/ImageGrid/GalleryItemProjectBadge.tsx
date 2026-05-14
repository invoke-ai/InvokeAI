import { Flex, Icon } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { PiStackBold } from 'react-icons/pi';

/**
 * Centered stack-icon badge laid over a canvas project thumbnail in the gallery grid. Purely
 * visual — distinguishes saved Canvas Projects from regular images/videos at a glance.
 */
export const GalleryItemProjectBadge = memo(() => {
  return (
    <Flex
      position="absolute"
      top="50%"
      left="50%"
      transform="translate(-50%, -50%)"
      pointerEvents="none"
      alignItems="center"
      justifyContent="center"
      bg="blackAlpha.500"
      borderRadius="full"
      boxSize={10}
      sx={{
        '@container (max-width: 80px)': {
          boxSize: 7,
        },
      }}
    >
      <Icon as={PiStackBold} boxSize="60%" color="white" filter="drop-shadow(0 0 2px rgba(0,0,0,0.6))" />
    </Flex>
  );
});

GalleryItemProjectBadge.displayName = 'GalleryItemProjectBadge';
