import { Badge, Box } from '@chakra-ui/react';
import { FindInGalleryThumbnailButton } from '@features/gallery/mediaSlot';
import { memo, useEffect, useRef } from 'react';

/**
 * Seek to frame midpoints to avoid rounding into adjacent frames. Keep one video element per source to avoid a
 * range fetch on every drag tick.
 */

const PREVIEW_VIDEO_STYLE = {
  display: 'block',
  height: '100%',
  objectFit: 'contain',
  width: '100%',
} as const;

export const TrimBoundThumb = memo(function TrimBoundThumb({
  fps,
  frame,
  label,
  name,
  src,
  onFindInGallery,
}: {
  fps: number;
  frame: number;
  label: string;
  /** The clip's file name, which names the find control among its siblings. */
  name?: string;
  src: string;
  /** Reveals the clip these bounds are cut from; omitted, the thumb is display only. */
  onFindInGallery?: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const element = videoRef.current;
    const time = Math.max(0, (frame + 0.5) / fps);

    if (!element || !Number.isFinite(time)) {
      return;
    }

    const seek = () => {
      element.currentTime = time;
    };

    // Seeking before metadata arrives is ignored by some browsers; wait for it
    // once, then seek directly on every later bound change.
    if (element.readyState >= 1) {
      seek();
    } else {
      element.addEventListener('loadedmetadata', seek, { once: true });

      return () => element.removeEventListener('loadedmetadata', seek);
    }
  }, [fps, frame, src]);

  return (
    <Box
      bg="blackAlpha.300"
      className="group"
      flexShrink={0}
      h="14"
      overflow="hidden"
      position="relative"
      rounded="sm"
      w="20"
    >
      <video key={src} ref={videoRef} muted preload="metadata" src={src} style={PREVIEW_VIDEO_STYLE} />
      <Badge bottom="0.5" insetInlineStart="0.5" pointerEvents="none" position="absolute" size="xs" variant="solid">
        {label}
      </Badge>
      {onFindInGallery ? <FindInGalleryThumbnailButton name={name} onFind={onFindInGallery} /> : null}
    </Box>
  );
});
