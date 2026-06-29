import { Icon, type SystemStyleObject } from '@chakra-ui/react';
import { absolutizeApiUrl } from '@workbench/backend/http';
import { StreamingImageFrame } from '@workbench/images/StreamingImageFrame';
import { imageUrlToStreamingSource, progressImageToStreamingSource } from '@workbench/images/streamingImageSource';
import { ImageOffIcon } from 'lucide-react';

/**
 * Square result preview for a queue item. Falls back to a deterministic gradient
 * derived from the item id, so items without a (yet) resolvable image still read
 * as distinct tiles — and the real thumbnail cross-fades in once loaded. A
 * hairline inset outline (theme border token) gives the tile consistent edge
 * depth without picking up the surface color underneath.
 */
const QUEUE_ITEM_THUMBNAIL_SX: SystemStyleObject = {
  boxShadow: 'inset 0 0 0 1px {colors.border.subtle}',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
} as const;
const IMAGE_OFF_PLACEHOLDER = <Icon as={ImageOffIcon} w="4" h="4" />;

export const QueueItemThumbnail = ({
  imageName,
  liveImage,
  // seed,
  boxSize = '8',
  rounded = 'md',
}: {
  imageName: string | null;
  liveImage?: { dataUrl: string; width: number; height: number } | null;
  // seed: number;
  boxSize?: string;
  rounded?: string;
}) => {
  const finalImage = imageUrlToStreamingSource({
    alt: imageName ?? '',
    src: imageName ? absolutizeApiUrl(`/api/v1/images/i/${encodeURIComponent(imageName)}/thumbnail`) : null,
  });

  return (
    <StreamingImageFrame
      bg="bg.subtle"
      boxSize={boxSize}
      flexShrink={0}
      finalImage={finalImage}
      liveImage={progressImageToStreamingSource(liveImage)}
      rounded={rounded}
      css={QUEUE_ITEM_THUMBNAIL_SX}
    >
      {IMAGE_OFF_PLACEHOLDER}
    </StreamingImageFrame>
  );
};
