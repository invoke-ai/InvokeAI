import type { GenerateReferenceImageAsset } from '@features/generation/core/types';
import type { LucideIcon } from 'lucide-react';
import type { CSSProperties, PointerEvent, RefCallback } from 'react';

import { Box, createListCollection, HStack, Portal, Spacer, Stack, Text } from '@chakra-ui/react';
import { galleryTransfers } from '@features/gallery';
import {
  constrainReferenceCropBox,
  FULL_REFERENCE_IMAGE_CROP_BOX,
  generatedImageToReferenceImage,
  getReferenceImageCropBoxPct,
  isFullReferenceImageCropBox,
  matchReferenceCropAspectRatio,
  REFERENCE_CROP_ASPECT_RATIOS,
  resolveReferenceCropDrag,
  resolveReferenceImageCrop,
  type ReferenceCropAspectRatioId,
  type ReferenceCropDrag,
  type ReferenceCropDragMode,
  type ReferenceCropHandle,
  type ReferenceImageCropBoxPct,
} from '@features/generation/core/referenceImage';
import { getReferenceImageUrls } from '@features/generation/data/referenceImageUrls';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { Button, CloseButton, IconButton } from '@platform/ui';
import { Dialog } from '@platform/ui/Dialog';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import { MaximizeIcon, ScanIcon, ZoomInIcon, ZoomOutIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Matches the legacy cropper's minimum crop dimension, in image pixels. */
const MIN_CROP_DIMENSION_PX = 64;

/** Zoom is relative to the fitted image; the legacy cropper's wheel and button steps apply. */
const ZOOM_WHEEL_FACTOR = 1.1;
const ZOOM_BUTTON_FACTOR = 1.2;
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 16;

const HANDLES: { cursor: string; left: string; name: ReferenceCropHandle; top: string }[] = [
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

const ASPECT_RATIO_TRIGGER_PROPS = { minW: '7rem' } as const;

const CONTAIN_IMG_STYLE: CSSProperties = { height: '100%', objectFit: 'contain', width: '100%' };

/** Fit: the image fills the viewport; pan is in viewport pixels. */
interface CropView {
  panX: number;
  panY: number;
  zoom: number;
}

const FIT_VIEW: CropView = { panX: 0, panY: 0, zoom: 1 };

type PanState = 'idle' | 'ready' | 'panning';

const isDragMode = (value: unknown): value is ReferenceCropDragMode =>
  value === 'move' || HANDLES.some((handle) => handle.name === value);

const isAspectRatioId = (value: unknown): value is ReferenceCropAspectRatioId =>
  REFERENCE_CROP_ASPECT_RATIOS.some((preset) => preset.id === value);

/** Space pans only when nothing that activates on Space has focus; buttons and fields keep their own behaviour. */
const consumesSpace = (target: EventTarget | null): boolean =>
  target instanceof Element &&
  target.closest(
    'button, [role="button"], input, textarea, select, [contenteditable], [role="combobox"], [role="listbox"]'
  ) !== null;

/** A saved partial crop reopens on its preset; the full image opens free, so the first drag is unconstrained. */
const initialAspectRatioId = (image: GenerateReferenceImageAsset): ReferenceCropAspectRatioId => {
  const box = getReferenceImageCropBoxPct(image);
  return isFullReferenceImageCropBox(box) ? 'free' : matchReferenceCropAspectRatio(box, image.original.image);
};

/** Pointer position as a percentage of the (transformed) stage, which holds the image at its on-screen size. */
const getPointerPct = (event: PointerEvent, stage: HTMLElement): { x: number; y: number } => {
  const rect = stage.getBoundingClientRect();

  return {
    x: ((event.clientX - rect.left) / rect.width) * 100,
    y: ((event.clientY - rect.top) / rect.height) * 100,
  };
};

const CropViewButton = ({ icon: Icon, label, onClick }: { icon: LucideIcon; label: string; onClick: () => void }) => (
  <Tooltip content={label}>
    <IconButton aria-label={label} size="sm" variant="ghost" onClick={onClick}>
      <Icon />
    </IconButton>
  </Tooltip>
);

/** Capture keeps a drag alive outside the viewport; without it (synthetic pointers) the viewport still sees moves. */
const capturePointer = (element: HTMLElement, pointerId: number): void => {
  try {
    element.setPointerCapture(pointerId);
  } catch {
    // No capture available for this pointer.
  }
};

/** Zooms by `factor` about a viewport point, so the pixel under the cursor stays put. */
const zoomViewAt = (view: CropView, factor: number, point: { x: number; y: number }): CropView => {
  const zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, view.zoom * factor));
  const scale = zoom / view.zoom;

  return {
    panX: point.x - (point.x - view.panX) * scale,
    panY: point.y - (point.y - view.panY) * scale,
    zoom,
  };
};

export const exportCroppedReferenceImage = async (
  image: GenerateReferenceImageAsset,
  cropBox: ReferenceImageCropBoxPct,
  owner: AccountScope = captureAccountScope()
): Promise<File> => {
  assertAccountScopeCurrent(owner);
  const original = image.original.image;
  const img = new Image();
  img.crossOrigin = 'anonymous';

  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('Failed to load reference image for cropping.'));
    img.src = getReferenceImageUrls({ original: image.original }).imageUrl;
  });

  assertAccountScopeCurrent(owner);
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

  assertAccountScopeCurrent(owner);
  return new File([blob], `${original.image_name.replace(/\.[^.]+$/, '')}-crop.png`, { type: 'image/png' });
};

export const applyReferenceImageCropSelection = async (
  image: GenerateReferenceImageAsset,
  cropBox: ReferenceImageCropBoxPct,
  owner: AccountScope = captureAccountScope()
): Promise<GenerateReferenceImageAsset> => {
  assertAccountScopeCurrent(owner);

  if (isFullReferenceImageCropBox(cropBox)) {
    return resolveReferenceImageCrop(image, cropBox, null);
  }

  const file = await exportCroppedReferenceImage(image, cropBox, owner);
  assertAccountScopeCurrent(owner);
  const croppedImage = await galleryTransfers.upload(file, 'none', {
    isIntermediate: true,
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);
  return resolveReferenceImageCrop(image, cropBox, generatedImageToReferenceImage(croppedImage).original.image);
};

export const ReferenceImageCropDialog = ({
  image,
  isOpen,
  onApply,
  onClose,
  onExitComplete,
}: {
  image: GenerateReferenceImageAsset;
  isOpen: boolean;
  onApply: (image: GenerateReferenceImageAsset) => void;
  onClose: () => void;
  /** After the close animation; the host releases the retained dialog here. */
  onExitComplete?: () => void;
}) => {
  const { t } = useTranslation();
  const { gallery, notifications } = useGenerationUi();
  const original = image.original.image;
  const [cropBox, setCropBox] = useState<ReferenceImageCropBoxPct>(() => getReferenceImageCropBoxPct(image));
  const [aspectRatioId, setAspectRatioId] = useState<ReferenceCropAspectRatioId>(() => initialAspectRatioId(image));
  const [view, setView] = useState<CropView>(FIT_VIEW);
  const [panState, setPanState] = useState<PanState>('idle');
  const [isApplying, setIsApplying] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<ReferenceCropDrag | null>(null);
  const panRef = useRef<{ panX: number; panY: number; x: number; y: number } | null>(null);
  const spaceHeldRef = useRef(false);
  const imageUrl = getReferenceImageUrls({ original: image.original }).imageUrl;
  const ratio = original.height > 0 ? original.width / original.height : 1;
  const constraints = useMemo(
    () => ({
      image: original,
      minDimensionPx: MIN_CROP_DIMENSION_PX,
      ratio: REFERENCE_CROP_ASPECT_RATIOS.find((preset) => preset.id === aspectRatioId)?.ratio ?? null,
    }),
    [aspectRatioId, original]
  );
  const cropWidthPx = Math.max(1, Math.round((cropBox.width / 100) * original.width));
  const cropHeightPx = Math.max(1, Math.round((cropBox.height / 100) * original.height));
  const isBusy = isApplying || isSaving;

  const aspectRatioCollection = useMemo(
    () =>
      createListCollection({
        items: REFERENCE_CROP_ASPECT_RATIOS.map(({ id }) => ({
          label: id === 'free' ? t('widgets.generate.cropAspectRatioFree') : id,
          value: id,
        })),
      }),
    [t]
  );

  const cropBoxStyle = useMemo<CSSProperties>(
    () => ({
      height: `${cropBox.height}%`,
      left: `${cropBox.x}%`,
      top: `${cropBox.y}%`,
      width: `${cropBox.width}%`,
    }),
    [cropBox]
  );
  const stageStyle = useMemo<CSSProperties>(
    () => ({ transform: `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`, transformOrigin: '0 0' }),
    [view]
  );

  const close = useCallback(() => {
    if (!isBusy) {
      setCropBox(getReferenceImageCropBoxPct(image));
      onClose();
    }
  }, [image, isBusy, onClose]);

  const handleOpenChange = useCallback(
    (event: Dialog.OpenChangeDetails) => {
      if (!event.open) {
        close();
      }
    },
    [close]
  );

  // Wheel zoom must cancel the dialog's own scrolling, which React's passive wheel listener cannot; Space is read
  // window-wide so panning works wherever focus sits, except in fields that use Space themselves.
  const viewportCallbackRef = useCallback<RefCallback<HTMLDivElement>>((element) => {
    viewportRef.current = element;
    if (!element) {
      return;
    }
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = element.getBoundingClientRect();
      const factor = event.deltaY < 0 ? ZOOM_WHEEL_FACTOR : 1 / ZOOM_WHEEL_FACTOR;
      setView((current) => zoomViewAt(current, factor, { x: event.clientX - rect.left, y: event.clientY - rect.top }));
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !consumesSpace(event.target)) {
        event.preventDefault();
        if (!event.repeat) {
          spaceHeldRef.current = true;
          setPanState((current) => (current === 'idle' ? 'ready' : current));
        }
      }
    };
    const releaseSpace = (event: KeyboardEvent | FocusEvent) => {
      if (!('code' in event) || event.code === 'Space') {
        spaceHeldRef.current = false;
        setPanState((current) => (current === 'ready' ? 'idle' : current));
      }
    };
    element.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', releaseSpace);
    window.addEventListener('blur', releaseSpace);
    return () => {
      element.removeEventListener('wheel', onWheel);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', releaseSpace);
      window.removeEventListener('blur', releaseSpace);
    };
  }, []);

  const zoomAboutCenter = useCallback((factor: number) => {
    const viewport = viewportRef.current;
    if (viewport) {
      setView((current) => zoomViewAt(current, factor, { x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 }));
    }
  }, []);
  const zoomIn = useCallback(() => zoomAboutCenter(ZOOM_BUTTON_FACTOR), [zoomAboutCenter]);
  const zoomOut = useCallback(() => zoomAboutCenter(1 / ZOOM_BUTTON_FACTOR), [zoomAboutCenter]);
  const fitView = useCallback(() => setView(FIT_VIEW), []);
  // Actual size: one image pixel per CSS pixel, centered.
  const resetView = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport || viewport.clientWidth === 0) {
      return;
    }
    const zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, original.width / viewport.clientWidth));
    setView({
      panX: (viewport.clientWidth * (1 - zoom)) / 2,
      panY: (viewport.clientHeight * (1 - zoom)) / 2,
      zoom,
    });
  }, [original.width]);

  const handleAspectRatioChange = useCallback(
    ({ value }: { value: string[] }) => {
      const next = value[0];
      if (!isAspectRatioId(next)) {
        return;
      }
      setAspectRatioId(next);
      setCropBox((current) =>
        constrainReferenceCropBox(current, {
          ...constraints,
          ratio: REFERENCE_CROP_ASPECT_RATIOS.find((preset) => preset.id === next)?.ratio ?? null,
        })
      );
    },
    [constraints]
  );

  const beginPan = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const viewport = viewportRef.current;
      if (!viewport || dragRef.current || (panState !== 'ready' && event.button !== 1)) {
        return;
      }
      event.preventDefault();
      capturePointer(viewport, event.pointerId);
      panRef.current = { panX: view.panX, panY: view.panY, x: event.clientX, y: event.clientY };
      setPanState('panning');
    },
    [panState, view.panX, view.panY]
  );

  const beginDrag = useCallback(
    (event: PointerEvent<HTMLElement>) => {
      const viewport = viewportRef.current;
      const stage = stageRef.current;
      const mode = (event.currentTarget as HTMLElement).dataset.dragMode;

      // Space or the middle button pans from anywhere, the crop box included; let it reach the viewport.
      if (!viewport || !stage || !isDragMode(mode) || panState === 'ready' || event.button !== 0) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      capturePointer(viewport, event.pointerId);
      dragRef.current = { box: cropBox, mode, ...getPointerPct(event, stage) };
    },
    [cropBox, panState]
  );

  const movePointer = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const pan = panRef.current;
      if (pan) {
        setView((current) => ({
          ...current,
          panX: pan.panX + event.clientX - pan.x,
          panY: pan.panY + event.clientY - pan.y,
        }));
        return;
      }
      const stage = stageRef.current;
      const drag = dragRef.current;
      if (stage && drag) {
        setCropBox(resolveReferenceCropDrag(drag, getPointerPct(event, stage), constraints));
      }
    },
    [constraints]
  );

  const endPointer = useCallback((event: PointerEvent<HTMLDivElement>) => {
    dragRef.current = null;
    if (panRef.current) {
      panRef.current = null;
      // Space still held keeps the viewport ready for the next pan.
      setPanState(spaceHeldRef.current ? 'ready' : 'idle');
    }
    if (viewportRef.current?.hasPointerCapture(event.pointerId)) {
      viewportRef.current.releasePointerCapture(event.pointerId);
    }
  }, []);

  const resetCrop = useCallback(() => {
    setAspectRatioId('free');
    setCropBox(FULL_REFERENCE_IMAGE_CROP_BOX);
  }, []);

  const applyCrop = useCallback(async () => {
    const owner = captureAccountScope();
    const isFullCrop = isFullReferenceImageCropBox(cropBox);

    if (!isFullCrop) {
      setIsApplying(true);
    }

    try {
      const croppedImage = await applyReferenceImageCropSelection(image, cropBox, owner);

      assertAccountScopeCurrent(owner);
      if (!isFullCrop) {
        gallery.touchImages();
      }

      assertAccountScopeCurrent(owner);
      onApply(croppedImage);
      if (isFullCrop) {
        close();
      } else {
        onClose();
      }
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      notifications.reportError({
        area: 'reference-images',
        message: error instanceof Error ? error.message : String(error),
        namespace: 'generation',
      });
    } finally {
      if (!isFullCrop && isAccountScopeCurrent(owner)) {
        setIsApplying(false);
      }
    }
  }, [close, cropBox, gallery, image, notifications, onApply, onClose]);

  // A durable copy of the crop in Uploads on the Gallery's auto-add board, unrelated to applying it: the dialog
  // stays open and the reference unchanged.
  const saveToAssets = useCallback(async () => {
    const owner = captureAccountScope();
    setIsSaving(true);

    try {
      const file = await exportCroppedReferenceImage(image, cropBox, owner);
      assertAccountScopeCurrent(owner);
      await galleryTransfers.upload(file, gallery.uploadBoardId, { isIntermediate: false, signal: owner.signal });
      assertAccountScopeCurrent(owner);
      gallery.touchImages();
      notifications.info(t('widgets.generate.cropSavedToAssets'));
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      notifications.reportError({
        area: 'reference-images',
        message: error instanceof Error ? error.message : String(error),
        namespace: 'generation',
      });
    } finally {
      if (isAccountScopeCurrent(owner)) {
        setIsSaving(false);
      }
    }
  }, [cropBox, gallery, image, notifications, t]);

  const aspectRatioValue = useMemo(() => [aspectRatioId], [aspectRatioId]);

  return (
    <Dialog.Root
      lazyMount
      open={isOpen}
      size="lg"
      unmountOnExit
      onExitComplete={onExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{t('widgets.generate.cropReferenceImage')}</Dialog.Title>
            </Dialog.Header>
            <Dialog.CloseTrigger asChild>
              <CloseButton disabled={isBusy} />
            </Dialog.CloseTrigger>
            <Dialog.Body>
              <Stack gap="3">
                <HStack gap="2">
                  <Text color="fg.muted" fontSize="sm">
                    {t('widgets.generate.aspectRatio')}
                  </Text>
                  <Select
                    aria-label={t('widgets.generate.aspectRatio')}
                    collection={aspectRatioCollection}
                    disabled={isBusy}
                    size="sm"
                    triggerProps={ASPECT_RATIO_TRIGGER_PROPS}
                    value={aspectRatioValue}
                    w="auto"
                    onValueChange={handleAspectRatioChange}
                  />
                  <Spacer />
                  <CropViewButton icon={MaximizeIcon} label={t('widgets.generate.cropFitView')} onClick={fitView} />
                  <CropViewButton icon={ScanIcon} label={t('widgets.generate.cropActualSize')} onClick={resetView} />
                  <CropViewButton icon={ZoomInIcon} label={t('widgets.generate.cropZoomIn')} onClick={zoomIn} />
                  <CropViewButton icon={ZoomOutIcon} label={t('widgets.generate.cropZoomOut')} onClick={zoomOut} />
                </HStack>
                <Box
                  ref={viewportCallbackRef}
                  aspectRatio={ratio}
                  bg="bg.muted"
                  cursor={panState === 'panning' ? 'grabbing' : panState === 'ready' ? 'grab' : undefined}
                  data-pan-state={panState}
                  maxH="60vh"
                  overflow="hidden"
                  position="relative"
                  rounded="md"
                  touchAction="none"
                  userSelect="none"
                  w="full"
                  onPointerCancel={endPointer}
                  onPointerDown={beginPan}
                  onPointerMove={movePointer}
                  onPointerUp={endPointer}
                >
                  <Box ref={stageRef} data-crop-stage="" inset="0" position="absolute" style={stageStyle}>
                    <img alt={original.image_name} draggable={false} src={imageUrl} style={CONTAIN_IMG_STYLE} />
                    <Box
                      borderColor="accent.solid"
                      borderWidth="1.5px"
                      boxShadow="0 0 0 9999px rgba(0, 0, 0, 0.45)"
                      cursor={panState === 'idle' ? 'move' : undefined}
                      data-crop-box=""
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
                          cursor={panState === 'idle' ? handle.cursor : undefined}
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
                </Box>
                <HStack gap="3">
                  <Text color="fg.muted" fontSize="sm">
                    {t('widgets.generate.cropReferenceImageHelp')}
                  </Text>
                  <Spacer />
                  <Text color="fg.muted" flexShrink={0} fontFamily="mono" fontSize="xs">
                    {cropWidthPx} × {cropHeightPx} px
                  </Text>
                </HStack>
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button disabled={isApplying} loading={isSaving} variant="outline" onClick={saveToAssets}>
                {t('widgets.generate.cropSaveToAssets')}
              </Button>
              <Spacer />
              <Button disabled={isBusy || isFullReferenceImageCropBox(cropBox)} variant="outline" onClick={resetCrop}>
                {t('common.reset')}
              </Button>
              <Button disabled={isBusy} variant="ghost" onClick={close}>
                {t('common.cancel')}
              </Button>
              <Button disabled={isSaving} loading={isApplying} onClick={applyCrop}>
                {t('common.apply')}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
