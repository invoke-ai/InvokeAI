import { Box, Flex, Icon, Image } from '@chakra-ui/react';
import { FolderIcon } from 'lucide-react';
import { useCallback, useState } from 'react';

/**
 * Reserve cover geometry to prevent reflow. Missing or deleted covers use the permanent glyph; images are
 * decorative beside the project name.
 */

export const PROJECT_COVER_ASPECT_RATIO = 16 / 10;

export const ProjectCover = ({ coverUrl }: { coverUrl?: string }) => {
  const [hasFailed, setHasFailed] = useState(false);
  const handleError = useCallback(() => setHasFailed(true), []);
  const showImage = Boolean(coverUrl) && !hasFailed;

  return (
    <Box aspectRatio={PROJECT_COVER_ASPECT_RATIO} bg="bg.muted" overflow="hidden" position="relative" w="full">
      {showImage ? (
        <Image alt="" h="full" objectFit="cover" src={coverUrl} w="full" onError={handleError} />
      ) : (
        <Flex align="center" h="full" justify="center" w="full">
          <Icon aria-hidden as={FolderIcon} boxSize="7" color="fg.subtle" opacity={0.6} />
        </Flex>
      )}
    </Box>
  );
};
