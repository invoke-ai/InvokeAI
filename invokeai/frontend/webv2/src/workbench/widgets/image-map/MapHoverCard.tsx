import type { GalleryItemKey } from '@features/gallery/contracts';
import type { ImageMapClusterLabelInfo, ImageMapImageLabels } from '@workbench/image-map/api';
import type { HoverThumbnail } from '@workbench/image-map/thumbnailCache';

import { Badge, Box, chakra, HStack, Stack, Text } from '@chakra-ui/react';
import { formatGalleryVideoDuration, parseGalleryItemKey } from '@features/gallery/contracts';
import { getClusterColor } from '@workbench/image-map/clusterPalette';
import { getImageLabels } from '@workbench/image-map/imageLabelCache';
import { PlayIcon } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

/** Bounds the hover card's thumbnail. */
const HOVER_PREVIEW_MAX_PX = 160;
const HOVER_PREVIEW_OFFSET_PX = 14;
/** Keep the hover card at least this clear of the viewport edges. */
const HOVER_PREVIEW_EDGE_PAD_PX = 10;

export interface HoverPreview {
  /** The hovered point's gallery key; the card parses it for the name and kind. */
  key: GalleryItemKey;
  thumbnail: HoverThumbnail;
  clientX: number;
  clientY: number;
}

/**
 * Resolve hovered cluster identity from current points; live reclustering may renumber ids under a stationary
 * pointer.
 */
export interface HoverCluster {
  /** DBSCAN cluster of the hovered point; -1 means unclustered noise. */
  cluster: number;
  /** Points currently on the map in that cluster. */
  clusterSize: number;
}

/** "Label: a, b, c" with the first (primary) tag emphasized. */
const HoverTagsRow = ({ prefix, tags }: { prefix: string; tags: string[] }) => (
  <Text color="fg.muted" fontSize="xs">
    <chakra.span color="fg.subtle">{prefix}</chakra.span>
    {tags.map((tag, index) => (
      <chakra.span key={tag} fontWeight={index === 0 ? '600' : undefined}>
        {index > 0 ? ', ' : ''}
        {tag}
      </chakra.span>
    ))}
  </Text>
);

/**
 * Measure asynchronous hover-card content invisibly before cursor-relative placement and viewport flipping. Key by
 * item for clean state and label video thumbnails explicitly.
 */
export const MapHoverCard = ({
  preview,
  hoverCluster,
  clusterLabel,
}: {
  preview: HoverPreview;
  hoverCluster: HoverCluster;
  clusterLabel: ImageMapClusterLabelInfo | null;
}) => {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [imageLabels, setImageLabels] = useState<ImageMapImageLabels | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);

  // Fetch tags lazily by stable item key and use session caching; DTO refresh identity must not retrigger
  // requests.
  const item = parseGalleryItemKey(preview.key);

  useEffect(() => {
    let cancelled = false;

    void getImageLabels(parseGalleryItemKey(preview.key)).then((labels) => {
      if (!cancelled) {
        setImageLabels(labels);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [preview.key]);

  // Re-measure whenever content that changes the card's size lands.
  useLayoutEffect(() => {
    const card = cardRef.current;

    if (!card) {
      return;
    }

    const rect = card.getBoundingClientRect();
    let left = preview.clientX + HOVER_PREVIEW_OFFSET_PX;
    let top = preview.clientY + HOVER_PREVIEW_OFFSET_PX;

    if (left + rect.width > window.innerWidth - HOVER_PREVIEW_EDGE_PAD_PX) {
      left = Math.max(0, preview.clientX - rect.width - HOVER_PREVIEW_OFFSET_PX);
    }

    if (top + rect.height > window.innerHeight - HOVER_PREVIEW_EDGE_PAD_PX) {
      top = Math.max(0, preview.clientY - rect.height - HOVER_PREVIEW_OFFSET_PX);
    }

    setPosition({ left, top });
  }, [preview.clientX, preview.clientY, imageLabels, imageLoaded, clusterLabel, hoverCluster]);

  const handleImageSettled = useCallback(() => setImageLoaded(true), []);
  const videoDuration =
    preview.thumbnail.durationSeconds === null ? null : formatGalleryVideoDuration(preview.thumbnail.durationSeconds);
  const clusterColor = getClusterColor(hoverCluster.cluster);
  const clusterTags = clusterLabel ? [clusterLabel.label, ...clusterLabel.alternates].slice(0, 3) : null;
  const imageTags = imageLabels ? [imageLabels.label, ...imageLabels.alternates].slice(0, 3) : null;

  return (
    <Stack
      bg="bg.muted"
      borderColor="border.emphasized"
      borderWidth="1px"
      color="fg"
      gap="1"
      left={`${position?.left ?? 0}px`}
      maxW="60"
      p="2"
      pointerEvents="none"
      position="fixed"
      ref={cardRef}
      rounded="md"
      shadow="lg"
      top={`${position?.top ?? 0}px`}
      visibility={position ? 'visible' : 'hidden'}
      zIndex="tooltip"
    >
      <chakra.img
        alt={item.kind === 'video' ? `Video: ${item.name}` : item.name}
        display="block"
        maxH={`${HOVER_PREVIEW_MAX_PX}px`}
        maxW={`${HOVER_PREVIEW_MAX_PX}px`}
        mx="auto"
        onError={handleImageSettled}
        onLoad={handleImageSettled}
        rounded="l2"
        src={preview.thumbnail.url}
      />
      <HStack gap="1.5">
        <Box bg={clusterColor} boxSize="2" flexShrink={0} rounded="full" />
        <Text fontSize="xs" fontWeight="600">
          {hoverCluster.cluster < 0
            ? 'Unclustered'
            : `Cluster ${hoverCluster.cluster} · ${hoverCluster.clusterSize} items`}
        </Text>
      </HStack>
      <HStack alignItems="flex-start" gap="1.5">
        {videoDuration !== null ? (
          <Badge display="flex" flexShrink={0} fontVariantNumeric="tabular-nums" gap="1" size="xs" variant="solid">
            <PlayIcon aria-hidden="true" fill="currentColor" />
            {videoDuration}
          </Badge>
        ) : null}
        <Text color="fg.muted" fontSize="xs" wordBreak="break-all">
          {item.name}
        </Text>
      </HStack>
      {clusterTags ? <HoverTagsRow prefix="Cluster tags: " tags={clusterTags} /> : null}
      {imageTags ? <HoverTagsRow prefix="Image tags: " tags={imageTags} /> : null}
    </Stack>
  );
};
