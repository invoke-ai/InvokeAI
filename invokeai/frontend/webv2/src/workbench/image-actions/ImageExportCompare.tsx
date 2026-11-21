import type { ImageSize } from '@features/gallery/contracts';
import type { CSSProperties, KeyboardEvent, PointerEvent } from 'react';

import { Box, HStack, Stack, Text } from '@chakra-ui/react';
import { LoadError } from '@platform/ui/LoadError';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

type ExportZoom = 'fit' | 'actual';

interface Pan {
  x: number;
  y: number;
}

/** While nothing has loaded, the frame holds a common photo shape instead of collapsing. */
const PLACEHOLDER_SIZE: ImageSize = { height: 3, width: 4 };
const DIVIDER_KEY_STEP = 5;

const LAYER_STYLE: CSSProperties = { inset: 0, overflow: 'hidden', pointerEvents: 'none', position: 'absolute' };
const FIT_IMAGE_STYLE: CSSProperties = {
  display: 'block',
  height: '100%',
  left: 0,
  position: 'absolute',
  top: 0,
  userSelect: 'none',
  width: '100%',
};
const FRAME_TOUCH_STYLE: CSSProperties = { touchAction: 'none' };
const NO_TRANSITION = { transition: 'none' } as const;
const DIVIDER_FOCUS_RING = {
  outline: 'none',
  '& > span': { boxShadow: '0 0 0 2px var(--chakra-colors-accent-solid)' },
} as const;

const clampPan = (pan: Pan, frame: ImageSize, image: ImageSize): Pan => {
  const clampAxis = (value: number, frameLength: number, imageLength: number) =>
    imageLength <= frameLength
      ? (frameLength - imageLength) / 2
      : Math.min(0, Math.max(frameLength - imageLength, value));

  return { x: clampAxis(pan.x, frame.width, image.width), y: clampAxis(pan.y, frame.height, image.height) };
};

const getFrameSize = (frame: HTMLElement | null): ImageSize => {
  const rect = frame?.getBoundingClientRect();

  return { height: rect?.height ?? 0, width: rect?.width ?? 0 };
};

/** The original against the export, split by a divider; at 100% both show at the original's pixels and pan together. */
export const ImageExportCompare = ({
  error,
  exportBlob,
  exportLabel,
  isEncoding,
  isRetrying,
  originalAlt,
  originalUrl,
  sourceSize,
  onExportLoad,
  onOriginalLoad,
  onRetry,
}: {
  error: { detail: string | null; message: string } | null;
  exportBlob: Blob | null;
  exportLabel: string;
  isEncoding: boolean;
  isRetrying: boolean;
  originalAlt: string;
  originalUrl: string;
  /** The original's pixel size; until known, the frame keeps a placeholder shape. */
  sourceSize: ImageSize | null;
  onExportLoad: (size: ImageSize) => void;
  onOriginalLoad: (size: ImageSize) => void;
  onRetry: () => void;
}) => {
  const { t } = useTranslation();
  const [split, setSplit] = useState(50);
  const [zoom, setZoom] = useState<ExportZoom>('fit');
  const [pan, setPan] = useState<Pan>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ kind: 'divider' } | { kind: 'pan'; origin: Pan; x: number; y: number } | null>(null);
  const shape = sourceSize ?? PLACEHOLDER_SIZE;
  const isActual = zoom === 'actual' && sourceSize !== null;

  const exportImageRef = useCallback(
    (node: HTMLImageElement | null) => {
      if (!node || !exportBlob) {
        return;
      }

      const url = URL.createObjectURL(exportBlob);

      node.src = url;
      return () => URL.revokeObjectURL(url);
    },
    [exportBlob]
  );

  const setSplitFromPointer = useCallback((clientX: number) => {
    const rect = frameRef.current?.getBoundingClientRect();

    if (rect && rect.width > 0) {
      setSplit(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
    }
  }, []);

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (event.pointerType === 'mouse' && event.button !== 0) {
        return;
      }

      const onDivider = event.target instanceof Element && event.target.closest('[data-export-divider]') !== null;

      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      if (isActual && !onDivider) {
        dragRef.current = { kind: 'pan', origin: pan, x: event.clientX, y: event.clientY };
        setIsPanning(true);
        return;
      }

      dragRef.current = { kind: 'divider' };
      setSplitFromPointer(event.clientX);
    },
    [isActual, pan, setSplitFromPointer]
  );

  const handlePointerMove = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current;

      if (!drag) {
        return;
      }
      if (drag.kind === 'divider') {
        setSplitFromPointer(event.clientX);
        return;
      }
      if (sourceSize) {
        setPan(
          clampPan(
            { x: drag.origin.x + event.clientX - drag.x, y: drag.origin.y + event.clientY - drag.y },
            getFrameSize(frameRef.current),
            sourceSize
          )
        );
      }
    },
    [setSplitFromPointer, sourceSize]
  );

  const endDrag = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    setIsPanning(false);
  }, []);

  const handleDividerKeyDown = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? DIVIDER_KEY_STEP * 2 : DIVIDER_KEY_STEP;
    const next =
      event.key === 'ArrowLeft' || event.key === 'ArrowDown'
        ? (value: number) => value - step
        : event.key === 'ArrowRight' || event.key === 'ArrowUp'
          ? (value: number) => value + step
          : event.key === 'Home'
            ? () => 0
            : event.key === 'End'
              ? () => 100
              : null;

    if (next) {
      event.preventDefault();
      setSplit((value) => Math.min(100, Math.max(0, next(value))));
    }
  }, []);

  const handleZoomChange = useCallback(
    (value: string) => {
      const nextZoom: ExportZoom = value === 'actual' ? 'actual' : 'fit';

      setZoom(nextZoom);
      if (nextZoom === 'actual' && sourceSize) {
        const frame = getFrameSize(frameRef.current);
        const fraction = split / 100;

        // Keep the point under the divider under it, where the two sides meet.
        setPan(
          clampPan(
            { x: fraction * (frame.width - sourceSize.width), y: (frame.height - sourceSize.height) / 2 },
            frame,
            sourceSize
          )
        );
      }
    },
    [sourceSize, split]
  );

  const handleOriginalLoad = useCallback(
    (event: { currentTarget: HTMLImageElement }) =>
      onOriginalLoad({ height: event.currentTarget.naturalHeight, width: event.currentTarget.naturalWidth }),
    [onOriginalLoad]
  );
  const handleExportLoad = useCallback(
    (event: { currentTarget: HTMLImageElement }) =>
      onExportLoad({ height: event.currentTarget.naturalHeight, width: event.currentTarget.naturalWidth }),
    [onExportLoad]
  );

  const imageStyle = useMemo<CSSProperties>(
    () =>
      isActual && sourceSize
        ? {
            display: 'block',
            height: sourceSize.height,
            left: 0,
            maxWidth: 'none',
            position: 'absolute',
            top: 0,
            transform: `translate(${pan.x}px, ${pan.y}px)`,
            userSelect: 'none',
            width: sourceSize.width,
          }
        : FIT_IMAGE_STYLE,
    [isActual, pan.x, pan.y, sourceSize]
  );
  const exportLayerStyle = useMemo<CSSProperties>(
    () => ({ ...LAYER_STYLE, clipPath: `inset(0 0 0 ${split}%)` }),
    [split]
  );
  const zoomOptions = useMemo(
    () => [
      { label: t('widgets.gallery.export.zoomFit'), value: 'fit' },
      { disabled: sourceSize === null, label: t('widgets.gallery.export.zoomActual'), value: 'actual' },
    ],
    [sourceSize, t]
  );

  return (
    <Stack bg="bg.inset" gap="0" h="full" minH="0" minW="0">
      <Box containerType="size" flex="1" minH="56" p="3" position="relative">
        <Box
          ref={frameRef}
          aspectRatio={`${shape.width} / ${shape.height}`}
          cursor={isActual ? (isPanning ? 'grabbing' : 'grab') : 'ew-resize'}
          insetInlineStart="50%"
          layerStyle="dotGrid"
          outline="1px solid"
          outlineColor="border.subtle"
          overflow="hidden"
          position="absolute"
          style={FRAME_TOUCH_STYLE}
          top="50%"
          transform="translate(-50%, -50%)"
          w={`min(calc(100cqw - var(--chakra-spacing-6)), calc((100cqh - var(--chakra-spacing-6)) * ${shape.width / shape.height}))`}
          onLostPointerCapture={endDrag}
          onPointerCancel={endDrag}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={endDrag}
        >
          <div style={LAYER_STYLE}>
            <img alt={originalAlt} draggable={false} src={originalUrl} style={imageStyle} onLoad={handleOriginalLoad} />
          </div>
          {exportBlob ? (
            <Box
              _motionReduce={NO_TRANSITION}
              opacity={isEncoding ? 0.6 : 1}
              style={exportLayerStyle}
              transitionDuration="moderate"
              transitionProperty="opacity"
            >
              <img
                ref={exportImageRef}
                alt={exportLabel}
                draggable={false}
                style={imageStyle}
                onLoad={handleExportLoad}
              />
            </Box>
          ) : null}
          <CompareTag insetInlineStart="1.5">{t('widgets.gallery.export.original')}</CompareTag>
          <CompareTag color="accent.fg" insetInlineEnd="1.5">
            {exportLabel}
          </CompareTag>
          <Box
            aria-label={t('widgets.gallery.export.compareDivider')}
            aria-valuemax={100}
            aria-valuemin={0}
            aria-valuenow={Math.round(split)}
            aria-valuetext={t('widgets.gallery.export.compareDividerValue', { percent: Math.round(split) })}
            bottom="0"
            cursor="ew-resize"
            data-export-divider=""
            insetInlineStart={`${split}%`}
            ms="-3"
            position="absolute"
            role="slider"
            tabIndex={0}
            top="0"
            w="6"
            _focusVisible={DIVIDER_FOCUS_RING}
            onKeyDown={handleDividerKeyDown}
          >
            <Box
              bg="fg"
              bottom="0"
              boxShadow="0 0 0 1px rgb(0 0 0 / 0.35)"
              left="calc(50% - 1px)"
              position="absolute"
              top="0"
              w="2px"
            />
            <Box
              as="span"
              bg="fg"
              borderRadius="full"
              h="5"
              left="50%"
              position="absolute"
              top="50%"
              transform="translate(-50%, -50%)"
              w="2.5"
            />
          </Box>
        </Box>
        {/* Outside the frame, whose pointer capture would swallow the Retry click. */}
        {error ? (
          <Box bg="bg.panel" borderRadius="md" bottom="4" boxShadow="md" insetInline="4" p="2" position="absolute">
            <LoadError
              detail={error.detail}
              isRetrying={isRetrying}
              layout="inline"
              message={error.message}
              retryLabel={t('widgets.gallery.export.retryPreview')}
              onRetry={onRetry}
            />
          </Box>
        ) : null}
      </Box>
      <HStack borderColor="border.subtle" borderTopWidth="1px" gap="2" px="3" py="1.5">
        <SegmentedControl
          ariaLabel={t('widgets.gallery.export.zoom')}
          isFullWidth={false}
          options={zoomOptions}
          size="xs"
          value={isActual ? 'actual' : 'fit'}
          onChange={handleZoomChange}
        />
        <Text aria-live="polite" color="fg.muted" fontSize="sm" ms="auto" role="status">
          {isEncoding ? t('widgets.gallery.export.encoding') : ''}
        </Text>
      </HStack>
    </Stack>
  );
};

const CompareTag = ({ children, ...props }: Parameters<typeof Text>[0]) => (
  <Text
    bg="bg.panel/80"
    borderRadius="sm"
    color="fg"
    fontSize="xs"
    fontWeight="600"
    pointerEvents="none"
    position="absolute"
    px="1.5"
    py="0.5"
    top="1.5"
    whiteSpace="nowrap"
    {...props}
  >
    {children}
  </Text>
);
