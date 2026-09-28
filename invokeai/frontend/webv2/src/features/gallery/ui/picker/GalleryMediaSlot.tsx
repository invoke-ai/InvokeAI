import type { DragEndEvent } from '@dnd-kit/core';
import type { GalleryItem, GalleryItemKind, GalleryItemRef } from '@features/gallery/core/items';

import { Box, HStack, Icon, Image, Spinner, Stack, Text } from '@chakra-ui/react';
import { useDndMonitor } from '@dnd-kit/core';
import { classifyGalleryUpload, getGalleryUploadAccept } from '@features/gallery/core/items';
import { getGalleryItemByRef } from '@features/gallery/data/backend';
import { getGalleryImageThumbnailUrl } from '@features/gallery/data/imageUrls';
import { getGalleryVideoThumbnailUrl } from '@features/gallery/data/videoUrls';
import { FindInGalleryThumbnailButton } from '@features/gallery/ui/FindInGalleryButton';
import { isGalleryItemDragData, useGalleryItemDroppable } from '@features/gallery/ui/galleryDnd';
import { useGalleryUi } from '@features/gallery/ui/GalleryUiContext';
import { useGalleryUploadInput } from '@features/gallery/ui/useGalleryUploadInput';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { DropTargetOverlay } from '@platform/ui/DropTargetOverlay';
import { DropZone } from '@platform/ui/DropZone';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ChevronDownIcon, ImagePlusIcon, RefreshCwIcon, UploadIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryPickerAccept } from './galleryPicker';

import { GalleryPickerPopover } from './GalleryPickerPopover';

const DROP_ZONE_FOCUS_PROPS = {
  outlineColor: 'accent.focusRing',
  outlineOffset: '2px',
  outlineStyle: 'solid',
  outlineWidth: '2px',
};
const DROP_ZONE_HOVER_PROPS = { bg: 'bg.muted', color: 'fg' };
const DROP_ZONE_DISABLED_PROPS = { cursor: 'not-allowed', opacity: 0.6 };

export interface GalleryMediaSlotValue {
  height?: number;
  kind: GalleryItemKind;
  name: string;
  width?: number;
}

export interface GalleryMediaSlotLabels {
  choose: string;
  /** Overlay call-to-action while a compatible drag is in flight. */
  drop: string;
  remove: string;
  replace: string;
}

const getDefaultLabels = (accept: GalleryPickerAccept, t: (key: string) => string): GalleryMediaSlotLabels => {
  const noun = accept.length === 1 && accept[0] === 'video' ? 'Video' : accept.length === 1 ? 'Image' : 'Media';

  return {
    choose: t(`widgets.gallery.picker.choose${noun}`),
    drop: t(`widgets.gallery.picker.drop${noun}`),
    remove: t(`widgets.gallery.picker.remove${noun}`),
    replace: t(`widgets.gallery.picker.replace${noun}`),
  };
};

const getThumbnailUrl = (value: GalleryMediaSlotValue): string =>
  value.kind === 'video' ? getGalleryVideoThumbnailUrl(value.name) : getGalleryImageThumbnailUrl(value.name);

/**
 * Own async resolution and report complete items through onChange. onUploadFile and custom thumbnails support
 * media stored outside the gallery.
 */
export const GalleryMediaSlot = ({
  accept,
  disabled = false,
  disabledReason,
  dropId,
  labels: labelOverrides,
  thumbnail,
  value,
  onChange,
  onFind,
  onUploadFile,
}: {
  accept: GalleryPickerAccept;
  disabled?: boolean;
  /** Shown in place of the affordances while disabled (e.g. a mutual exclusion). */
  disabledReason?: string;
  /** Unique droppable id; sibling slots must not share one. */
  dropId: string;
  labels?: Partial<GalleryMediaSlotLabels>;
  /** Replaces the gallery thumbnail of `value`, for media the gallery does not hold. */
  thumbnail?: ReactNode;
  value: GalleryMediaSlotValue | null;
  onChange: (item: GalleryItem | null) => void;
  /** Provide onFind only for gallery-owned media; it enables the thumbnail reveal badge. */
  onFind?: () => void;
  /** Takes a file the consumer stores itself; gallery-backed slots upload through the picker instead. */
  onUploadFile?: (file: File) => void;
}) => {
  const { t } = useTranslation();
  const { notifications } = useGalleryUi();
  const [isBusy, setIsBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const isInert = disabled || isBusy;
  const labels = useMemo(() => ({ ...getDefaultLabels(accept, t), ...labelOverrides }), [accept, labelOverrides, t]);

  // Advertise single drags of accepted kinds, but stay armed for any drag of
  // those kinds: a multi-item release must be a dead drop, not fall through.
  const { acceptsDrag, shieldsDrag } = useMemo(
    () => ({
      acceptsDrag: (data: unknown) =>
        isGalleryItemDragData(data) && data.items.length === 1 && accept.includes(data.items[0]!.kind),
      shieldsDrag: (data: unknown) =>
        isGalleryItemDragData(data) && data.items.some((ref) => accept.includes(ref.kind)),
    }),
    [accept]
  );
  const { acceptsActiveDrag, isOver, setNodeRef } = useGalleryItemDroppable(
    acceptsDrag,
    { data: { kind: dropId }, disabled: isInert, id: dropId },
    shieldsDrag
  );

  const fail = useCallback(
    (error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);

      setErrorMessage(message);
      notifications.reportError({ area: 'gallery-media-slot', message, namespace: 'gallery' });
    },
    [notifications]
  );

  const adoptRef = useCallback(
    async (ref: GalleryItemRef) => {
      const owner = captureAccountScope();

      setErrorMessage(null);
      setIsBusy(true);

      try {
        const item = await getGalleryItemByRef(ref, owner.signal);

        assertAccountScopeCurrent(owner);
        onChange(item);
      } catch (error) {
        if (isAccountScopeCurrent(owner)) {
          fail(error);
        }
      } finally {
        if (isAccountScopeCurrent(owner)) {
          setIsBusy(false);
        }
      }
    },
    [fail, onChange]
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const data = event.active.data.current;

      if (!isInert && event.over?.id === dropId && acceptsDrag(data) && isGalleryItemDragData(data)) {
        void adoptRef(data.items[0]!);
      }
    },
    [acceptsDrag, adoptRef, dropId, isInert]
  );

  useDndMonitor({ onDragEnd: handleDragEnd });

  const uploadOptions = useMemo(() => ({ accept: getGalleryUploadAccept(accept), multiple: false }), [accept]);
  const handleUpload = useCallback(
    ([file]: File[]) => {
      const kind = file ? classifyGalleryUpload(file)?.kind : undefined;

      setErrorMessage(null);

      // The input's `accept` is advisory ("All files" bypasses it); a kind the
      // slot cannot take must not be handed on only to be dropped on the floor.
      if (!file || (kind && !accept.includes(kind))) {
        setErrorMessage(
          t(kind === 'video' ? 'widgets.gallery.picker.unsupportedVideo' : 'widgets.gallery.picker.unsupportedImage')
        );
        return;
      }

      onUploadFile?.(file);
    },
    [accept, onUploadFile, t]
  );
  const { inputProps: uploadInputProps, openPicker: openUploadPicker } = useGalleryUploadInput(
    handleUpload,
    uploadOptions
  );
  const handleClear = useCallback(() => onChange(null), [onChange]);
  const handlePick = useCallback(
    (item: GalleryItem) => {
      setErrorMessage(null);
      onChange(item);
    },
    [onChange]
  );

  return (
    <Stack gap="2">
      <Box ref={setNodeRef} className="group" position="relative">
        <GalleryPickerPopover accept={accept} label={labels.choose} onPick={handlePick}>
          <DropZone
            as="button"
            aria-busy={isBusy || undefined}
            aria-disabled={disabled || undefined}
            aria-label={value ? labels.replace : labels.choose}
            cursor={disabled ? 'not-allowed' : undefined}
            disabled={isInert}
            isDisabled={isInert}
            isInvalid={errorMessage !== null}
            isOver={isOver}
            minH="20"
            overflow="hidden"
            textAlign="start"
            w="full"
            _disabled={DROP_ZONE_DISABLED_PROPS}
            _focusVisible={DROP_ZONE_FOCUS_PROPS}
            _hover={isInert ? undefined : DROP_ZONE_HOVER_PROPS}
          >
            {value ? (
              <HStack align="stretch" gap="3" h="20" p="2">
                <Box bg="blackAlpha.300" boxSize="16" flexShrink="0" overflow="hidden" rounded="sm">
                  {thumbnail ?? (
                    <Image
                      alt=""
                      boxSize="full"
                      objectFit="contain"
                      outline="1px solid"
                      outlineColor="border.image"
                      outlineOffset="-1px"
                      rounded="sm"
                      src={getThumbnailUrl(value)}
                    />
                  )}
                </Box>
                <Stack align="start" flex="1" gap="1" justify="center" minW="0">
                  <MiddleTruncate color="fg" fontSize="xs" fontWeight="semibold" text={value.name} />
                  {value.width && value.height ? (
                    <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums">
                      {value.width} × {value.height}
                    </Text>
                  ) : null}
                  <HStack color="fg.muted" gap="1">
                    {isBusy ? <Spinner size="xs" /> : <Icon as={RefreshCwIcon} boxSize="2.5" />}
                    <Text fontSize="2xs">
                      {disabled && disabledReason
                        ? disabledReason
                        : isBusy
                          ? t('widgets.gallery.picker.working')
                          : t('widgets.gallery.picker.replaceHint')}
                    </Text>
                  </HStack>
                </Stack>
              </HStack>
            ) : (
              <Stack align="center" color="fg.muted" gap="1.5" justify="center" minH="20" px="4">
                {isBusy ? (
                  <Spinner size="sm" />
                ) : (
                  <HStack color="fg" fontSize="xs" fontWeight="600" gap="1.5">
                    <Icon as={ImagePlusIcon} boxSize="4" />
                    {labels.choose}
                    <Icon as={ChevronDownIcon} boxSize="3" color="fg.subtle" />
                  </HStack>
                )}
                <Text color="fg.muted" fontSize="2xs" textAlign="center">
                  {disabled && disabledReason
                    ? disabledReason
                    : isBusy
                      ? t('widgets.gallery.picker.working')
                      : t('widgets.gallery.picker.dropHint')}
                </Text>
              </Stack>
            )}
          </DropZone>
        </GalleryPickerPopover>
        {value && onFind ? (
          /*
           * Keep the badge beside the button face to avoid nested buttons; matching row metrics align it without
           * measurement.
           */
          <HStack gap="3" h="20" insetInline="0" p="2" pointerEvents="none" position="absolute" top="0">
            <Box boxSize="16" flexShrink="0" position="relative">
              <FindInGalleryThumbnailButton name={value.name} onFind={onFind} />
            </Box>
          </HStack>
        ) : null}
        <DropTargetOverlay isActive={acceptsActiveDrag} isOver={isOver} label={labels.drop} />
      </Box>
      {(onUploadFile && !disabled) || value ? (
        <HStack justify="end">
          {onUploadFile && !disabled ? (
            <Button disabled={isBusy} size="xs" variant="ghost" onClick={openUploadPicker}>
              <Icon as={UploadIcon} boxSize="3" />
              {t('widgets.gallery.picker.upload')}
            </Button>
          ) : null}
          {value ? (
            <Button disabled={isBusy} size="xs" variant="ghost" onClick={handleClear}>
              <Icon as={XIcon} boxSize="3" />
              {labels.remove}
            </Button>
          ) : null}
        </HStack>
      ) : null}
      {errorMessage ? (
        <Text aria-live="polite" color="fg.error" fontSize="2xs" role="alert" textWrap="pretty">
          {errorMessage}
        </Text>
      ) : null}
      {onUploadFile ? <input {...uploadInputProps} /> : null}
    </Stack>
  );
};
