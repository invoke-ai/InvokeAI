import type { DragEndEvent } from '@dnd-kit/core';
import type { ImageWithDims } from '@features/generation/contracts';
import type { ChangeEvent } from 'react';

import { Box, HStack, Image, Input, Spinner, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor } from '@dnd-kit/core';
import { galleryImages, galleryTransfers } from '@features/gallery';
import { galleryImageUrls, isGalleryImageDragData, useGalleryImageDroppable } from '@features/gallery/utility';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { DropZone } from '@platform/ui/DropZone';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ImagePlusIcon, XIcon } from 'lucide-react';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useVideoUiActions } from './VideoUiContext';

/**
 * A conditioning keyframe (first or last frame): gallery drop target, file
 * upload, and the chosen image's row. Same async-ownership pattern as the
 * Upscale input image — this is the section that owns loading/error state.
 */

const UPLOAD_ACCEPT_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const UPLOAD_ACCEPT_ATTR = [...UPLOAD_ACCEPT_TYPES, '.png', '.jpg', '.jpeg', '.webp'].join(',');
const DROP_ZONE_FOCUS_PROPS = {
  outlineColor: 'accent.focusRing',
  outlineOffset: '2px',
  outlineStyle: 'solid',
  outlineWidth: '2px',
};
const DROP_ZONE_DISABLED_PROPS = { cursor: 'not-allowed', opacity: 0.6 };
const DROP_ZONE_BUSY_PROPS = { disabled: true };
const DROP_ZONE_HOVER_PROPS = { bg: 'bg.muted', color: 'fg' };

const areFrameImagesEquivalent = (left: ImageWithDims | null, right: ImageWithDims | null): boolean =>
  (left === null && right === null) || left?.image_name === right?.image_name;

export const VideoFrameImageField = memo(
  function VideoFrameImageField({
    disabled = false,
    disabledReason,
    dropId,
    image,
    onChange,
  }: {
    disabled?: boolean;
    /** Shown in place of the field's affordances while disabled (mutual exclusion). */
    disabledReason?: string;
    /** Unique droppable id — the first- and last-frame fields render side by side. */
    dropId: string;
    image: ImageWithDims | null;
    onChange: (image: ImageWithDims | null) => void;
  }) {
    const { t } = useTranslation();
    const { reportError, touchGalleryImages } = useVideoUiActions();
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const isInert = disabled || isLoading;
    const { isOver, setNodeRef } = useGalleryImageDroppable({
      data: { kind: dropId },
      disabled: isInert,
      id: dropId,
    });

    const setGalleryImage = useCallback(
      async (imageName: string) => {
        setErrorMessage(null);
        setIsLoading(true);

        try {
          const [resolved] = await galleryImages.resolveMany([imageName]);

          if (resolved) {
            onChange({ height: resolved.height, image_name: resolved.imageName, width: resolved.width });
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

        if (!isInert && event.over?.id === dropId && isGalleryImageDragData(data) && data.items.length === 1) {
          const imageName = data.items[0]?.name;

          if (imageName) {
            void setGalleryImage(imageName);
          }
        }
      },
      [dropId, isInert, setGalleryImage]
    );

    useDndMonitor({ onDragEnd: handleDragEnd });

    const uploadFile = useCallback(
      async (file: File) => {
        setErrorMessage(null);

        if (!UPLOAD_ACCEPT_TYPES.includes(file.type)) {
          setErrorMessage(t('widgets.video.unsupportedImageFile'));
          reportError(t('widgets.video.unsupportedImageFile'));
          return;
        }

        const owner = captureAccountScope();
        setIsLoading(true);

        try {
          const uploaded = await galleryTransfers.upload(file, 'none', { signal: owner.signal });

          assertAccountScopeCurrent(owner);
          onChange({ height: uploaded.height, image_name: uploaded.imageName, width: uploaded.width });
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
    const handlePickFile = useCallback(() => {
      if (!isInert) {
        fileInputRef.current?.click();
      }
    }, [isInert]);
    const handleClear = useCallback(() => onChange(null), [onChange]);

    return (
      <Stack gap="2">
        <DropZone
          ref={setNodeRef}
          as="button"
          aria-busy={isLoading}
          aria-disabled={disabled}
          aria-label={image ? t('widgets.video.replaceFrame') : t('widgets.video.uploadFrame')}
          cursor={disabled ? 'not-allowed' : 'pointer'}
          isOver={isOver}
          {...(isLoading ? DROP_ZONE_BUSY_PROPS : undefined)}
          minH="20"
          overflow="hidden"
          position="relative"
          _focusVisible={DROP_ZONE_FOCUS_PROPS}
          _disabled={DROP_ZONE_DISABLED_PROPS}
          _hover={isInert ? undefined : DROP_ZONE_HOVER_PROPS}
          opacity={disabled ? 0.6 : undefined}
          onClick={handlePickFile}
        >
          {image ? (
            <HStack align="stretch" gap="3" h="20" p="2">
              <Box bg="blackAlpha.300" boxSize="16" flexShrink="0" overflow="hidden" rounded="sm">
                <Image
                  alt=""
                  boxSize="full"
                  objectFit="contain"
                  outline="1px solid"
                  outlineColor="border.image"
                  outlineOffset="-1px"
                  rounded="sm"
                  src={galleryImageUrls.thumbnail(image.image_name)}
                />
              </Box>
              <Stack align="start" flex="1" gap="1" justify="center" minW="0">
                <MiddleTruncate color="fg" fontSize="xs" fontWeight="semibold" text={image.image_name} />
                <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums">
                  {image.width} × {image.height}
                </Text>
              </Stack>
            </HStack>
          ) : (
            <Stack align="center" color="fg.muted" gap="1.5" justify="center" minH="20" px="4">
              {isLoading ? <Spinner size="sm" /> : <ImagePlusIcon aria-hidden="true" size="18" />}
              <Text fontSize="2xs" textAlign="center">
                {disabled && disabledReason
                  ? disabledReason
                  : isLoading
                    ? t('widgets.video.uploadingFrame')
                    : t('widgets.video.uploadOrDropFrame')}
              </Text>
            </Stack>
          )}
        </DropZone>
        <HStack justify="space-between">
          {disabled && disabledReason && image ? (
            <Text color="fg.muted" fontSize="2xs">
              {disabledReason}
            </Text>
          ) : (
            <Box />
          )}
          {image ? (
            <Button disabled={isLoading} size="xs" variant="ghost" onClick={handleClear}>
              <XIcon aria-hidden="true" size="12" />
              {t('widgets.video.removeFrame')}
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
    previous.onChange === next.onChange &&
    previous.disabled === next.disabled &&
    previous.disabledReason === next.disabledReason &&
    previous.dropId === next.dropId &&
    areFrameImagesEquivalent(previous.image, next.image)
);
