import { Flex, Icon } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { PiPlayFill } from 'react-icons/pi';

/**
 * Centered play-button badge laid over a video thumbnail in the gallery grid. Purely visual —
 * the gallery item itself owns click selection; the play action lives in the viewer.
 */
export const GalleryItemPlayBadge = memo(() => {
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
      // Container-query-driven: shrink the badge on tiny thumbnails so it stays balanced.
      sx={{
        '@container (max-width: 80px)': {
          boxSize: 7,
        },
      }}
    >
      <Icon as={PiPlayFill} boxSize="60%" color="white" filter="drop-shadow(0 0 2px rgba(0,0,0,0.6))" />
    </Flex>
  );
});

GalleryItemPlayBadge.displayName = 'GalleryItemPlayBadge';
