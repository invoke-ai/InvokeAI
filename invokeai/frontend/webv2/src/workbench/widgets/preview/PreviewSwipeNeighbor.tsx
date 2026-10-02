import type { GalleryItem } from '@features/gallery';

import { Flex, Spinner } from '@chakra-ui/react';
import { useCallback, useState, type CSSProperties, type Ref } from 'react';

import type { PreviewNeighbor, PreviewNeighbors } from './usePreviewNavigation';

import { FittedFrame } from './PreviewStage';

/** Both neighbors of a swipeable stage, once the swipe has anything to show. */
export const PreviewSwipeNeighbors = ({
  neighbors,
  nextTrackRef,
  previousTrackRef,
  restStyle,
}: {
  neighbors: PreviewNeighbors;
  nextTrackRef: Ref<HTMLDivElement>;
  previousTrackRef: Ref<HTMLDivElement>;
  restStyle: CSSProperties;
}) => (
  <>
    <PreviewSwipeNeighbor
      neighbor={neighbors.previous}
      restStyle={restStyle}
      side="previous"
      trackRef={previousTrackRef}
    />
    <PreviewSwipeNeighbor neighbor={neighbors.next} restStyle={restStyle} side="next" trackRef={nextTrackRef} />
  </>
);

/**
 * One stage-width beside the preview, the item a swipe slides in. It shares the stage's padding and container
 * queries, so it fits exactly as the item will once selected; images use the full URL that navigation prefetches,
 * making the post-commit swap pixel-identical. An unloaded page shows a spinner while the step fetches it; a live
 * session shows the bare stage.
 */
const PreviewSwipeNeighbor = ({
  neighbor,
  restStyle,
  side,
  trackRef,
}: {
  neighbor: PreviewNeighbor;
  restStyle: CSSProperties;
  side: 'next' | 'previous';
  trackRef: Ref<HTMLDivElement>;
}) => (
  <Flex
    ref={trackRef}
    align="center"
    aria-hidden="true"
    bottom="0"
    css={NEIGHBOR_CSS}
    data-swipe-neighbor={side}
    justify="center"
    left={side === 'next' ? '100%' : '-100%'}
    pointerEvents="none"
    position="absolute"
    style={restStyle}
    top="0"
    w="full"
  >
    {neighbor?.kind === 'more' ? <Spinner color="fg.muted" size="md" /> : null}
    {neighbor?.kind === 'item' ? (
      // Keyed by URL: every layer is a new element per neighbor, so none can keep painting the previous picture while
      // its own loads.
      <NeighborMedia key={neighbor.item.fullUrl} item={neighbor.item} />
    ) : null}
  </Flex>
);

/**
 * The thumbnail (usually cached already) lies under the full image until that loads, so a swipe faster than the
 * download still reveals the right picture; it is removed once the full image arrives, leaving exactly what the
 * frame will show. No shadow: the neighbor sits just beyond the stage edge, where the frame shadow would bleed in.
 */
const NeighborMedia = ({ item }: { item: GalleryItem }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const handleSettled = useCallback(() => setIsLoaded(true), []);

  return (
    <FittedFrame
      bg={item.kind === 'video' ? 'black' : 'transparent'}
      boxShadow="none"
      frameHeight={item.height}
      frameWidth={item.width}
    >
      {item.kind === 'image' && !isLoaded ? (
        <img alt="" draggable={false} src={item.thumbnailUrl} style={THUMBNAIL_UNDERLAY_STYLE} />
      ) : null}
      <img
        ref={item.width * item.height <= DECODE_AHEAD_MAX_PIXELS ? decodeAhead : undefined}
        alt=""
        draggable={false}
        height={item.height}
        src={item.kind === 'video' ? item.thumbnailUrl : item.fullUrl}
        style={item.kind === 'video' ? VIDEO_POSTER_STYLE : IMAGE_STYLE}
        width={item.width}
        onError={handleSettled}
        onLoad={handleSettled}
      />
    </FittedFrame>
  );
};

/**
 * Largest neighbor decoded ahead of time. Above it (big upscales), the decoded bitmap would hold hundreds of MB per
 * neighbor for a swipe that may never come, so the image decodes when revealed instead.
 */
const DECODE_AHEAD_MAX_PIXELS = 4096 * 4096;

/** Off-stage images are never painted, so without a decode hint the first swipe frame could show them blank. */
const decodeAhead = (image: HTMLImageElement | null): void => {
  void image?.decode().catch(() => {});
};

// `inherit` copies the stage's padding, chrome inset included, onto this box.
const NEIGHBOR_CSS = { padding: 'inherit' } as const;

const IMAGE_STYLE: CSSProperties = { display: 'block', height: 'auto', position: 'relative', width: '100%' };

const THUMBNAIL_UNDERLAY_STYLE: CSSProperties = {
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  position: 'absolute',
  width: '100%',
};

const VIDEO_POSTER_STYLE: CSSProperties = { display: 'block', height: '100%', objectFit: 'contain', width: '100%' };
