import type { DragEndEvent } from '@dnd-kit/core';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';
import type { ChangeEvent } from 'react';

import { Box, HStack, Image, Input, Spinner, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor } from '@dnd-kit/core';
import { galleryImages, galleryTransfers } from '@features/gallery';
import {
  galleryImageUrls,
  isGalleryImageDragData,
  isSingleGalleryImageDragData,
  useGalleryItemDroppable,
} from '@features/gallery/utility';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { DropTargetOverlay } from '@platform/ui/DropTargetOverlay';
import { DropZone } from '@platform/ui/DropZone';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ImagePlusIcon, UploadIcon, XIcon } from 'lucide-react';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { areInputImagesEquivalent } from './upscaleComparators';
import { useUpscaleUiActions } from './UpscaleUiContext';

/**
 * The Upscale widget's source image: gallery drop target, file upload, and the
 * chosen image's row.
 *
 * Split from the widget body because it is the one section that owns async
 * work and local state -- resolving a dropped gallery item, uploading a file,
 * and the loading and error states both of those produce. Everything else in
 * the widget is a controlled field over project state, and keeping this beside
 * them made it easy to mistake its local `useState` for form state.
 */

const DROP_ID = 'upscale-input-image';
const UPLOAD_ACCEPT_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const UPLOAD_ACCEPT_ATTR = [...UPLOAD_ACCEPT_TYPES, '.png', '.jpg', '.jpeg', '.webp'].join(',');
const DROP_ZONE_FOCUS_PROPS = {
  outlineColor: 'accent.focusRing',
  outlineOffset: '2px',
  outlineStyle: 'solid',
  outlineWidth: '2px',
};
const DROP_ZONE_DISABLED_PROPS = { cursor: 'wait', opacity: 0.7 };
// `DropZone` types its props as `BoxProps`, which has no `disabled`; spreading a
// typed object keeps the cast in one place.
const DROP_ZONE_BUSY_PROPS = { disabled: true };
const DROP_ZONE_HOVER_PROPS = { bg: 'bg.muted', color: 'fg' };

export const UpscaleImageField = memo(
  function UpscaleImageField({
    inputImage,
    onChange,
  }: {
    inputImage: UpscaleWidgetValues['inputImage'];
    onChange: (image: UpscaleWidgetValues['inputImage']) => void;
  }) {
    const { t } = useTranslation();
    const { reportError, touchGalleryImages } = useUpscaleUiActions();
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    // Advertise single-image drags only (the drop handler below consumes
    // exactly one), but stay armed for ANY image drag: a multi-image release
    // here must remain a dead drop, not fall through to a target underneath.
    const { acceptsActiveDrag, isOver, setNodeRef } = useGalleryItemDroppable(
      isSingleGalleryImageDragData,
      { data: { kind: DROP_ID }, disabled: isLoading, id: DROP_ID },
      isGalleryImageDragData
    );

    const setGalleryImage = useCallback(
      async (imageName: string) => {
        setErrorMessage(null);
        setIsLoading(true);

        try {
          const [image] = await galleryImages.resolveMany([imageName]);

          if (image) {
            onChange({ height: image.height, image_name: image.imageName, width: image.width });
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          setErrorMessage(message);
          reportError(message);
        } finally {
          setIsLoading(false);
        }
      },
      [onChange, reportError]
    );

    const handleDragEnd = useCallback(
      (event: DragEndEvent) => {
        const data = event.active.data.current;

        if (!isLoading && event.over?.id === DROP_ID && isGalleryImageDragData(data) && data.items.length === 1) {
          const imageName = data.items[0]?.name;

          if (imageName) {
            void setGalleryImage(imageName);
          }
        }
      },
      [isLoading, setGalleryImage]
    );

    useDndMonitor({ onDragEnd: handleDragEnd });

    const uploadFile = useCallback(
      async (file: File) => {
        setErrorMessage(null);

        if (!UPLOAD_ACCEPT_TYPES.includes(file.type)) {
          setErrorMessage(t('widgets.upscale.unsupportedFile'));
          reportError(t('widgets.upscale.unsupportedFile'));
          return;
        }

        const owner = captureAccountScope();
        setIsLoading(true);

        try {
          const image = await galleryTransfers.upload(file, 'none', { signal: owner.signal });

          assertAccountScopeCurrent(owner);
          onChange({ height: image.height, image_name: image.imageName, width: image.width });
          touchGalleryImages();
        } catch (error) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          const message = error instanceof Error ? error.message : String(error);
          setErrorMessage(message);
          reportError(message);
        } finally {
          setIsLoading(false);
        }
      },
      [onChange, reportError, t, touchGalleryImages]
    );

    const handleFileChange = useCallback(
      (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.currentTarget.files?.[0];

        if (file) {
          void uploadFile(file);
        }
        event.currentTarget.value = '';
      },
      [uploadFile]
    );
    const handlePickFile = useCallback(() => fileInputRef.current?.click(), []);
    const handleClear = useCallback(() => onChange(null), [onChange]);

    return (
      <Stack gap="2">
        <DropZone
          ref={setNodeRef}
          as="button"
          aria-busy={isLoading}
          aria-label={inputImage ? t('widgets.upscale.replaceImage') : t('widgets.upscale.uploadImage')}
          cursor="pointer"
          isOver={isOver}
          {...(isLoading ? DROP_ZONE_BUSY_PROPS : undefined)}
          minH="24"
          overflow="hidden"
          position="relative"
          _focusVisible={DROP_ZONE_FOCUS_PROPS}
          _disabled={DROP_ZONE_DISABLED_PROPS}
          _hover={isLoading ? undefined : DROP_ZONE_HOVER_PROPS}
          onClick={handlePickFile}
        >
          {inputImage ? (
            <HStack align="stretch" gap="3" h="24" p="2">
              <Box bg="blackAlpha.300" boxSize="20" flexShrink="0" overflow="hidden" rounded="sm">
                <Image
                  alt={t('widgets.upscale.inputImageAlt')}
                  boxSize="full"
                  objectFit="contain"
                  outline="1px solid"
                  outlineColor="border.image"
                  outlineOffset="-1px"
                  rounded="sm"
                  src={galleryImageUrls.thumbnail(inputImage.image_name)}
                />
              </Box>
              <Stack align="start" flex="1" gap="1" justify="center" minW="0">
                <MiddleTruncate color="fg" fontSize="xs" fontWeight="semibold" text={inputImage.image_name} />
                <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums">
                  {inputImage.width} × {inputImage.height}
                </Text>
                <HStack color="fg.muted" gap="1">
                  {isLoading ? <Spinner size="xs" /> : <UploadIcon aria-hidden="true" size="12" />}
                  <Text fontSize="2xs">
                    {isLoading ? t('widgets.upscale.uploadingImage') : t('widgets.upscale.replaceOrDrop')}
                  </Text>
                </HStack>
              </Stack>
            </HStack>
          ) : (
            <Stack align="center" color="fg.muted" gap="2" justify="center" minH="24" px="4">
              {isLoading ? <Spinner size="sm" /> : <ImagePlusIcon aria-hidden="true" size="20" />}
              <Text fontSize="xs" textAlign="center">
                {isLoading ? t('widgets.upscale.uploadingImage') : t('widgets.upscale.uploadOrDrop')}
              </Text>
            </Stack>
          )}
          <DropTargetOverlay isActive={acceptsActiveDrag} isOver={isOver} label={t('widgets.upscale.dropImage')} />
        </DropZone>
        <HStack justify="end">
          {inputImage ? (
            <Button disabled={isLoading} size="xs" variant="ghost" onClick={handleClear}>
              <XIcon aria-hidden="true" size="12" />
              {t('widgets.upscale.removeImage')}
            </Button>
          ) : null}
        </HStack>
        {errorMessage ? (
          <Text aria-live="polite" color="fg.error" fontSize="2xs" role="alert" textWrap="pretty">
            {errorMessage}
          </Text>
        ) : null}
        <Input
          ref={fileInputRef}
          accept={UPLOAD_ACCEPT_ATTR}
          aria-hidden="true"
          display="none"
          tabIndex={-1}
          type="file"
          onChange={handleFileChange}
        />
      </Stack>
    );
  },
  (previous, next) =>
    previous.onChange === next.onChange && areInputImagesEquivalent(previous.inputImage, next.inputImage)
);
