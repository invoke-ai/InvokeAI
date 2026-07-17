/* eslint-disable react/react-compiler */
import type { StreamingImageSource } from '@workbench/images/streamingImageSource';

import { Badge, Box, Flex, type SystemStyleObject } from '@chakra-ui/react';
import { useCallback, useMemo, type CSSProperties, type MouseEvent, type ReactNode, type Ref } from 'react';
import { useTranslation } from 'react-i18next';

import { PreviewCompareDropZone } from './PreviewCompareDropZone';
import { PreviewLiveOverlay } from './PreviewLiveReadout';
import { usePreviewLoupe, type PreviewLoupeControls } from './usePreviewLoupe';

/**
 * The preview's image surface: a dot-grid backdrop with an aspect-fitted,
 * shadowed frame. Owns everything drawn over the image (live badge, and later
 * the loupe, progress hairline, and drop-to-compare affordance) so the widget
 * shell never grows frame-specific rendering.
 */

export const previewGridCss = {
  backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1.5px)',
  backgroundPosition: 'center',
  backgroundRepeat: 'repeat',
  backgroundSize: '24px 24px',
} as const;

export const getFittedFrameCss = (width: number, height: number): SystemStyleObject => ({
  aspectRatio: `${width} / ${height}`,
  height: 'auto',
  maxHeight: '100%',
  maxWidth: '100%',
  width: `min(100cqw, calc(100cqh * ${width / height}))`,
});

export const PreviewFrame = ({
  children,
  frameHeight,
  frameWidth,
  isLive,
  liveBadgeLabel,
  liveQueueItemId,
  loupeControlsRef,
  onContextMenu,
  padding,
  shouldAntialiasLiveImage,
  source,
  variant,
}: {
  /** Rendered instead of the fitted frame when there is no image source (inset variant only). */
  children?: ReactNode;
  frameHeight: number;
  frameWidth: number;
  isLive: boolean;
  liveBadgeLabel: string;
  /** When set, the static live badge is replaced by the live progress readout for this run. */
  liveQueueItemId?: string | null;
  /** Imperative zoom controls, for hotkeys registered by the widget shell. */
  loupeControlsRef?: Ref<PreviewLoupeControls>;
  onContextMenu?: (x: number, y: number) => void;
  padding?: string;
  shouldAntialiasLiveImage: boolean;
  source: StreamingImageSource | null;
  /** `framed` = bordered surface for a selected image; `inset` = flush surface for the empty state. */
  variant: 'framed' | 'inset';
}) => {
  const { t } = useTranslation();
  const loupe = usePreviewLoupe({
    controlsRef: loupeControlsRef,
    enabled: variant === 'framed' && !isLive,
    naturalWidth: frameWidth,
  });

  // Reset zoom in place when the displayed image changes (or goes live) — a
  // remount would flash the frame on every selection.
  loupe.syncDisplayedSource(variant === 'framed' && !isLive && source ? source.src : null);
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (onContextMenu) {
        event.preventDefault();
        onContextMenu(event.clientX, event.clientY);
      }
    },
    [onContextMenu]
  );
  const imageStyle = useMemo<CSSProperties>(
    () => ({
      display: 'block',
      height: 'auto',
      imageRendering: isLive && !shouldAntialiasLiveImage ? 'pixelated' : 'auto',
      width: '100%',
    }),
    [isLive, shouldAntialiasLiveImage]
  );
  const liveBadge = isLive ? (
    typeof liveQueueItemId === 'string' ? (
      <PreviewLiveOverlay queueItemId={liveQueueItemId} />
    ) : (
      <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
        {liveBadgeLabel}
      </Badge>
    )
  ) : null;

  if (variant === 'inset') {
    return (
      <Flex
        align="center"
        backgroundColor="bg.inset"
        color="fg.grid"
        containerType="size"
        css={previewGridCss}
        h="full"
        justify="center"
        w="full"
      >
        {source ? (
          <Box
            borderColor={isLive ? 'accent.solid' : 'border.emphasized'}
            borderWidth="1px"
            boxShadow="0 24px 80px rgba(0,0,0,0.42)"
            css={getFittedFrameCss(frameWidth, frameHeight)}
            overflow="hidden"
            position="relative"
            rounded="lg"
          >
            <img
              alt={source.alt}
              draggable={false}
              height={frameHeight}
              src={source.src}
              style={imageStyle}
              width={frameWidth}
            />
            {liveBadge}
          </Box>
        ) : (
          children
        )}
      </Flex>
    );
  }

  return (
    <Flex
      align="center"
      borderWidth="1px"
      borderColor="border.subtle"
      color="fg.grid"
      containerType="size"
      css={previewGridCss}
      flex="1"
      justify="center"
      minH="0"
      overflow="hidden"
      p={padding}
      position="relative"
      rounded="lg"
      w="full"
    >
      <PreviewCompareDropZone />
      <Box
        ref={loupe.frameRefCallback}
        bg="transparent"
        borderWidth="1px"
        borderColor={isLive ? 'accent.solid' : 'border.emphasized'}
        boxShadow="0 24px 80px rgba(0,0,0,0.42)"
        css={getFittedFrameCss(frameWidth, frameHeight)}
        cursor={loupe.isZoomed ? 'grab' : undefined}
        overflow="hidden"
        position="relative"
        rounded="lg"
        onContextMenu={onContextMenu ? handleContextMenu : undefined}
        {...loupe.frameProps}
      >
        {source ? (
          <img
            ref={loupe.imageRef}
            alt={source.alt}
            height={frameHeight}
            src={source.src}
            style={imageStyle}
            width={frameWidth}
          />
        ) : null}
        {liveBadge}
        {loupe.zoomPercent !== null ? (
          <Badge
            aria-label={t('widgets.preview.resetZoom')}
            as="button"
            bottom="2"
            cursor="pointer"
            position="absolute"
            right="2"
            size="xs"
            title={t('widgets.preview.resetZoom')}
            variant="solid"
            onClick={loupe.reset}
          >
            {loupe.zoomPercent}%
          </Badge>
        ) : null}
      </Box>
    </Flex>
  );
};
