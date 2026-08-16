import { Box, Flex, Icon, Image } from '@chakra-ui/react';
import { FolderIcon } from 'lucide-react';
import { useCallback, useState } from 'react';

/**
 * A project's thumbnail.
 *
 * `coverUrl` comes from the per-user cover index (`projects/covers.ts`), which
 * records each project's newest result as it saves. The box is reserved
 * unconditionally, so a card does not reflow when its cover resolves.
 *
 * The glyph is not a placeholder awaiting an image. It is the permanent state
 * for a project that has produced nothing, and for a cover whose image has
 * since been deleted — a broken `<img>` reads worse than the glyph, so a load
 * failure falls back to it.
 *
 * The image is decorative (`alt=""`): every caller renders the project's name
 * beside it, and announcing it twice helps nobody.
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
