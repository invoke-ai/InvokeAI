import type { GenerateReferenceImageAsset } from '@workbench/generation/types';
import type { CSSProperties, PointerEvent } from 'react';

import { Box, Dialog, Portal, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { uploadGalleryImage } from '@workbench/gallery/api';
import {
  FULL_REFERENCE_IMAGE_CROP_BOX,
  generatedImageToReferenceImage,
  getReferenceImageCropBoxPct,
  getReferenceImageUrls,
  isFullReferenceImageCropBox,
  resolveReferenceImageCrop,
  type ReferenceImageCropBoxPct,
} from '@workbench/generation/referenceImage';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

type HandleName = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

type DragMode = 'move' | HandleName;

type DragState = { box: ReferenceImageCropBoxPct; mode: DragMode; x: number; y: number };

/** Matches the legacy cropper's minimum crop dimension, in image pixels. */
const MIN_CROP_DIMENSION_PX = 64;

const HANDLES: { cursor: string; left: string; name: HandleName; top: string }[] = [
  { cursor: 'nwse-resize', left: '0%', name: 'nw', top: '0%' },
  { cursor: 'ns-resize', left: '50%', name: 'n', top: '0%' },
  { cursor: 'nesw-resize', left: '100%', name: 'ne', top: '0%' },
  { cursor: 'ew-resize', left: '100%', name: 'e', top: '50%' },
  { cursor: 'nwse-resize', left: '100%', name: 'se', top: '100%' },
  { cursor: 'ns-resize', left: '50%', name: 's', top: '100%' },
  { cursor: 'nesw-resize', left: '0%', name: 'sw', top: '100%' },
  { cursor: 'ew-resize', left: '0%', name: 'w', top: '50%' },
];

const THIRDS_OFFSETS = ['33.333%', '66.667%'];

const CONTAIN_IMG_STYLE: CSSProperties = { height: '100%', objectFit: 'contain', width: '100%' };

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const getPointerPct = (event: PointerEvent, element: HTMLElement): { x: number; y: number } => {
  const rect = element.getBoundingClientRect();

  return {
    x: ((event.clientX - rect.left) / rect.width) * 100,
    y: ((event.clientY - rect.top) / rect.height) * 100,
  };
};

const resolveDrag = (
  drag: DragState,
  point: { x: number; y: number },
  minWidth: number,
  minHeight: number
): ReferenceImageCropBoxPct => {
  const dx = point.x - drag.x;
  const dy = point.y - drag.y;
  const box = drag.box;

  if (drag.mode === 'move') {
    return {
      ...box,
      x: clamp(box.x + dx, 0, 100 - box.width),
      y: clamp(box.y + dy, 0, 100 - box.height),
    };
  }

  let left = box.x;
  let top = box.y;
  let right = box.x + box.width;
  let bottom = box.y + box.height;

  if (drag.mode.includes('w')) {
    left = clamp(left + dx, 0, right - minWidth);
  }

  if (drag.mode.includes('e')) {
    right = clamp(right + dx, left + minWidth, 100);
  }

  if (drag.mode.includes('n')) {
    top = clamp(top + dy, 0, bottom - minHeight);
  }

  if (drag.mode.includes('s')) {
    bottom = clamp(bottom + dy, top + minHeight, 100);
  }

  return { height: bottom - top, width: right - left, x: left, y: top };
};

const isDragMode = (value: unknown): value is DragMode =>
  value === 'move' || HANDLES.some((handle) => handle.name === value);

export const exportCroppedReferenceImage = async (
  image: GenerateReferenceImageAsset,
  cropBox: ReferenceImageCropBoxPct
): Promise<File> => {
  const original = image.original.image;
  const img = new Image();
  img.crossOrigin = 'anonymous';

  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('Failed to load reference image for cropping.'));
    img.src = getReferenceImageUrls({ original: image.original }).imageUrl;
  });

  const sourceWidth = img.naturalWidth || original.width;
  const sourceHeight = img.naturalHeight || original.height;
  const sx = Math.round((cropBox.x / 100) * sourceWidth);
  const sy = Math.round((cropBox.y / 100) * sourceHeight);
  const sw = Math.max(1, Math.round((cropBox.width / 100) * sourceWidth));
  const sh = Math.max(1, Math.round((cropBox.height / 100) * sourceHeight));
  const canvas = document.createElement('canvas');

  canvas.width = sw;
  canvas.height = sh;
  canvas.getContext('2d')?.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (result) => (result ? resolve(result) : reject(new Error('Failed to export cropped image.'))),
      'image/png'
    );
  });

  return new File([blob], `${original.image_name.replace(/\.[^.]+$/, '')}-crop.png`, { type: 'image/png' });
};

export const applyReferenceImageCropSelection = async (
  image: GenerateReferenceImageAsset,
  cropBox: ReferenceImageCropBoxPct
): Promise<GenerateReferenceImageAsset> => {
  if (isFullReferenceImageCropBox(cropBox)) {
    return resolveReferenceImageCrop(image, cropBox, null);
  }

  const file = await exportCroppedReferenceImage(image, cropBox);
  const croppedImage = await uploadGalleryImage(file, 'none', { isIntermediate: true });

  return resolveReferenceImageCrop(image, cropBox, generatedImageToReferenceImage(croppedImage).original.image);
};

export const ReferenceImageCropDialog = ({
  image,
  isOpen,
  onApply,
  onClose,
}: {
  image: GenerateReferenceImageAsset;
  isOpen: boolean;
  onApply: (image: GenerateReferenceImageAsset) => void;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const [cropBox, setCropBox] = useState<ReferenceImageCropBoxPct>(() => getReferenceImageCropBoxPct(image));
  const [isApplying, setIsApplying] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const original = image.original.image;
  const imageUrl = getReferenceImageUrls({ original: image.original }).imageUrl;
  const ratio = original.height > 0 ? original.width / original.height : 1;
  const minWidthPct = Math.min(100, (MIN_CROP_DIMENSION_PX / Math.max(1, original.width)) * 100);
  const minHeightPct = Math.min(100, (MIN_CROP_DIMENSION_PX / Math.max(1, original.height)) * 100);
  const cropWidthPx = Math.max(1, Math.round((cropBox.width / 100) * original.width));
  const cropHeightPx = Math.max(1, Math.round((cropBox.height / 100) * original.height));

  const cropBoxStyle = useMemo<CSSProperties>(
    () => ({
      height: `${cropBox.height}%`,
      left: `${cropBox.x}%`,
      top: `${cropBox.y}%`,
      width: `${cropBox.width}%`,
    }),
    [cropBox]
  );

  const close = useCallback(() => {
    if (!isApplying) {
      setCropBox(getReferenceImageCropBoxPct(image));
      onClose();
    }
  }, [image, isApplying, onClose]);

  const handleOpenChange = useCallback(
    (event: Dialog.OpenChangeDetails) => {
      if (!event.open) {
        close();
      }
    },
    [close]
  );

  const beginDrag = useCallback(
    (event: PointerEvent<HTMLElement>) => {
      const container = containerRef.current;
      const mode = (event.currentTarget as HTMLElement).dataset.dragMode;

      if (!container || !isDragMode(mode)) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      container.setPointerCapture(event.pointerId);
      dragRef.current = { box: cropBox, mode, ...getPointerPct(event, container) };
    },
    [cropBox]
  );

  const moveDrag = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const container = containerRef.current;
      const drag = dragRef.current;

      if (!container || !drag) {
        return;
      }

      setCropBox(resolveDrag(drag, getPointerPct(event, container), minWidthPct, minHeightPct));
    },
    [minHeightPct, minWidthPct]
  );

  const endDrag = useCallback((event: PointerEvent<HTMLDivElement>) => {
    dragRef.current = null;

    if (containerRef.current?.hasPointerCapture(event.pointerId)) {
      containerRef.current.releasePointerCapture(event.pointerId);
    }
  }, []);

  const resetCrop = useCallback(() => setCropBox(FULL_REFERENCE_IMAGE_CROP_BOX), []);

  const applyCrop = useCallback(async () => {
    if (isFullReferenceImageCropBox(cropBox)) {
      onApply(await applyReferenceImageCropSelection(image, cropBox));
      close();
      return;
    }

    try {
      setIsApplying(true);
      const croppedImage = await applyReferenceImageCropSelection(image, cropBox);
      dispatch({ type: 'touchGalleryImagesRefresh' });
      onApply(croppedImage);
      onClose();
    } catch (error) {
      dispatch({
        area: 'reference-images',
        message: error instanceof Error ? error.message : String(error),
        namespace: 'generation',
        type: 'recordError',
      });
    } finally {
      setIsApplying(false);
    }
  }, [close, cropBox, dispatch, image, onApply, onClose]);

  return (
    <Dialog.Root lazyMount open={isOpen} placement="center" size="lg" unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title fontSize="sm" fontWeight="700">
                {t('widgets.generate.cropReferenceImage')}
              </Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="3">
                <Text color="fg.muted" fontSize="xs">
                  {t('widgets.generate.cropReferenceImageHelp')}
                </Text>
                <Box
                  ref={containerRef}
                  aspectRatio={ratio}
                  bg="bg.muted"
                  maxH="60vh"
                  overflow="hidden"
                  position="relative"
                  rounded="md"
                  touchAction="none"
                  userSelect="none"
                  w="full"
                  onPointerCancel={endDrag}
                  onPointerMove={moveDrag}
                  onPointerUp={endDrag}
                >
                  <img alt={original.image_name} draggable={false} src={imageUrl} style={CONTAIN_IMG_STYLE} />
                  <Box
                    borderColor="accent.solid"
                    borderWidth="1.5px"
                    boxShadow="0 0 0 9999px rgba(0, 0, 0, 0.45)"
                    cursor="move"
                    data-drag-mode="move"
                    position="absolute"
                    touchAction="none"
                    style={cropBoxStyle}
                    onPointerDown={beginDrag}
                  >
                    {THIRDS_OFFSETS.map((offset) => (
                      <Box
                        key={`v-${offset}`}
                        bg="whiteAlpha.500"
                        bottom="0"
                        left={offset}
                        pointerEvents="none"
                        position="absolute"
                        top="0"
                        w="1px"
                      />
                    ))}
                    {THIRDS_OFFSETS.map((offset) => (
                      <Box
                        key={`h-${offset}`}
                        bg="whiteAlpha.500"
                        h="1px"
                        left="0"
                        pointerEvents="none"
                        position="absolute"
                        right="0"
                        top={offset}
                      />
                    ))}
                    {HANDLES.map((handle) => (
                      <Box
                        key={handle.name}
                        bg="white"
                        boxShadow="0 0 0 1px rgba(0, 0, 0, 0.4), 0 1px 3px rgba(0, 0, 0, 0.5)"
                        cursor={handle.cursor}
                        data-drag-mode={handle.name}
                        h="3"
                        left={handle.left}
                        position="absolute"
                        rounded="xs"
                        top={handle.top}
                        touchAction="none"
                        transform="translate(-50%, -50%)"
                        w="3"
                        onPointerDown={beginDrag}
                      />
                    ))}
                  </Box>
                </Box>
                <Text color="fg.muted" fontFamily="mono" fontSize="2xs">
                  {cropWidthPx} × {cropHeightPx} px
                </Text>
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button
                disabled={isApplying || isFullReferenceImageCropBox(cropBox)}
                size="sm"
                variant="outline"
                onClick={resetCrop}
              >
                {t('common.reset')}
              </Button>
              <Button disabled={isApplying} size="sm" variant="outline" onClick={close}>
                {t('common.cancel')}
              </Button>
              <Button loading={isApplying} size="sm" onClick={applyCrop}>
                {t('common.apply')}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
