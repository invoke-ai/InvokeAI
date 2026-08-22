import type { DragEndEvent } from '@dnd-kit/core';
import type { VideoSourceClip } from '@features/video/core/types';
import type { ChangeEvent } from 'react';

import { Badge, Box, HStack, Input, Spinner, Stack, Text } from '@chakra-ui/react';
import { useDndContext, useDndMonitor, useDroppable } from '@dnd-kit/core';
import { galleryItems, galleryTransfers } from '@features/gallery';
import { galleryVideoUrls, isGalleryItemDragData } from '@features/gallery/utility';
import { createVideoSourceClip } from '@features/video/core/settings';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Field } from '@platform/ui';
import { Button } from '@platform/ui/Button';
import { DropTargetOverlay } from '@platform/ui/DropTargetOverlay';
import { DropZone } from '@platform/ui/DropZone';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import { FilmIcon, XIcon } from 'lucide-react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useVideoUiActions } from './VideoUiContext';

/**
 * The clip to extend: gallery drop target, file upload, a live frame preview,
 * and the trim bounds. The preview is a muted `<video>` element seeked to the
 * most recently adjusted trim bound (`frame / fps`) — browsers display the
 * frame natively without a canvas roundtrip, same trick as the workflow
 * editor's frame scrubber.
 */

const DROP_ID = 'video-source-clip';
const DROP_ZONE_FOCUS_PROPS = {
  outlineColor: 'accent.focusRing',
  outlineOffset: '2px',
  outlineStyle: 'solid',
  outlineWidth: '2px',
};
const DROP_ZONE_DISABLED_PROPS = { cursor: 'not-allowed', opacity: 0.6 };
const DROP_ZONE_BUSY_PROPS = { disabled: true };
const DROP_ZONE_HOVER_PROPS = { bg: 'bg.muted', color: 'fg' };
const PREVIEW_VIDEO_STYLE = {
  display: 'block',
  height: '100%',
  objectFit: 'contain',
  width: '100%',
} as const;

/**
 * Small always-visible preview of one trim bound, so both ends of the kept
 * range stay in view while either slider moves (the large preview only follows
 * the last-touched bound). Same seek technique as the workflow editor's frame
 * scrubber: one long-lived muted element, half-frame nudge so rounding cannot
 * show the neighbouring frame.
 */
const TrimBoundThumb = memo(function TrimBoundThumb({
  fps,
  frame,
  label,
  src,
}: {
  fps: number;
  frame: number;
  label: string;
  src: string;
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
    <Box bg="blackAlpha.300" flex="1" h="16" minW="0" overflow="hidden" position="relative" rounded="sm">
      <video key={src} ref={videoRef} muted preload="metadata" src={src} style={PREVIEW_VIDEO_STYLE} />
      <Badge
        bottom="1"
        fontVariantNumeric="tabular-nums"
        insetInlineStart="1"
        pointerEvents="none"
        position="absolute"
        size="xs"
        variant="solid"
      >
        {`${label} · ${frame}`}
      </Badge>
    </Box>
  );
});

const getSingleVideoDragName = (data: unknown): string | null => {
  if (!isGalleryItemDragData(data) || data.items.length !== 1) {
    return null;
  }

  const item = data.items[0];

  return item?.kind === 'video' ? item.name : null;
};

const areSourceClipsEquivalent = (left: VideoSourceClip | null, right: VideoSourceClip | null): boolean =>
  (left === null && right === null) ||
  (left?.video_name === right?.video_name &&
    left?.startFrame === right?.startFrame &&
    left?.endFrame === right?.endFrame &&
    // The bound thumbnails seek at frame/fps, so a same-name clip hydrated
    // with a different probed rate must re-render.
    left?.fps === right?.fps);

export const VideoSourceClipField = memo(
  function VideoSourceClipField({
    disabled = false,
    disabledReason,
    onChange,
    sourceVideo,
  }: {
    disabled?: boolean;
    /** Shown in place of the field's affordances while disabled (mutual exclusion). */
    disabledReason?: string;
    onChange: (clip: VideoSourceClip | null) => void;
    sourceVideo: VideoSourceClip | null;
  }) {
    const { t } = useTranslation();
    const { reportError, touchGalleryImages } = useVideoUiActions();
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    // Which trim bound the preview follows; the last one the user moved.
    const [previewBound, setPreviewBound] = useState<'start' | 'end'>('end');
    const isInert = disabled || isLoading;

    const { active } = useDndContext();
    const acceptsActiveDrag = !isInert && getSingleVideoDragName(active?.data.current) !== null;
    const { isOver, setNodeRef } = useDroppable({ disabled: !acceptsActiveDrag, id: DROP_ID });

    const adoptVideo = useCallback(
      async (videoName: string) => {
        setErrorMessage(null);
        setIsLoading(true);

        try {
          const item = await galleryItems.resolve({ kind: 'video', name: videoName });

          if (item?.kind === 'video') {
            onChange(
              createVideoSourceClip({
                durationSeconds: item.durationSeconds,
                fps: item.fps,
                height: item.height,
                name: item.name,
                width: item.width,
              })
            );
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
        const videoName = getSingleVideoDragName(event.active.data.current);

        if (!isInert && event.over?.id === DROP_ID && videoName) {
          void adoptVideo(videoName);
        }
      },
      [adoptVideo, isInert]
    );

    useDndMonitor({ onDragEnd: handleDragEnd });

    const uploadFile = useCallback(
      async (file: File) => {
        setErrorMessage(null);

        // `accept` is advisory; an unknown type (empty string) is the server's call.
        if (file.type && !file.type.startsWith('video/')) {
          setErrorMessage(t('widgets.video.unsupportedVideoFile'));
          reportError(t('widgets.video.unsupportedVideoFile'));
          return;
        }

        const owner = captureAccountScope();
        setIsLoading(true);

        try {
          const uploaded = await galleryTransfers.uploadVideo(file, 'none', { signal: owner.signal });

          assertAccountScopeCurrent(owner);
          onChange(
            createVideoSourceClip({
              durationSeconds: uploaded.durationSeconds,
              fps: uploaded.fps,
              height: uploaded.height,
              name: uploaded.name,
              width: uploaded.width,
            })
          );
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

    const maxFrameIndex = sourceVideo ? Math.max(0, sourceVideo.numFrames - 1) : 0;
    // The crossfade join needs a 2-frame tail from the trimmed source, so the
    // setters keep at least two frames between the bounds.
    const setStartFrame = useCallback(
      (rawStart: number) => {
        if (!sourceVideo) {
          return;
        }

        const startFrame = Math.min(Math.max(0, rawStart), Math.max(0, maxFrameIndex - 1));

        setPreviewBound('start');
        onChange({ ...sourceVideo, endFrame: Math.max(startFrame + 1, sourceVideo.endFrame), startFrame });
      },
      [maxFrameIndex, onChange, sourceVideo]
    );
    const setEndFrame = useCallback(
      (rawEnd: number) => {
        if (!sourceVideo) {
          return;
        }

        const endFrame = Math.max(1, Math.min(rawEnd, maxFrameIndex));

        setPreviewBound('end');
        onChange({ ...sourceVideo, endFrame, startFrame: Math.min(sourceVideo.startFrame, endFrame - 1) });
      },
      [maxFrameIndex, onChange, sourceVideo]
    );

    const previewFrame = sourceVideo ? (previewBound === 'start' ? sourceVideo.startFrame : sourceVideo.endFrame) : 0;
    // Seek to the middle of the frame's display interval so rounding cannot
    // show the neighbouring frame. Seeking sets currentTime on one long-lived
    // element (the frame-scrubber trick) — remounting per drag tick would spawn
    // a range fetch per movement on multi-MB clips.
    const previewTime = sourceVideo ? Math.max(0, (previewFrame + 0.5) / sourceVideo.fps) : 0;
    const previewVideoRef = useRef<HTMLVideoElement | null>(null);
    const previewSrc = sourceVideo ? galleryVideoUrls.full(sourceVideo.video_name) : null;

    useEffect(() => {
      const element = previewVideoRef.current;

      if (element && Number.isFinite(previewTime)) {
        element.currentTime = previewTime;
      }
    }, [previewSrc, previewTime]);

    return (
      <Stack gap="2">
        <DropZone
          ref={setNodeRef}
          as="button"
          aria-busy={isLoading}
          aria-disabled={disabled}
          aria-label={sourceVideo ? t('widgets.video.replaceClip') : t('widgets.video.uploadClip')}
          cursor={disabled ? 'not-allowed' : 'pointer'}
          isOver={isOver}
          {...(isLoading ? DROP_ZONE_BUSY_PROPS : undefined)}
          minH="24"
          overflow="hidden"
          position="relative"
          _focusVisible={DROP_ZONE_FOCUS_PROPS}
          _disabled={DROP_ZONE_DISABLED_PROPS}
          _hover={isInert ? undefined : DROP_ZONE_HOVER_PROPS}
          opacity={disabled ? 0.6 : undefined}
          onClick={handlePickFile}
        >
          {sourceVideo && previewSrc ? (
            <Stack gap="1" p="2">
              <Box bg="blackAlpha.300" h="28" overflow="hidden" rounded="sm" w="full">
                <video
                  key={previewSrc}
                  ref={previewVideoRef}
                  muted
                  preload="metadata"
                  src={previewSrc}
                  style={PREVIEW_VIDEO_STYLE}
                />
              </Box>
              <HStack justify="space-between" minW="0">
                <MiddleTruncate color="fg" flex="1" fontSize="xs" fontWeight="semibold" text={sourceVideo.video_name} />
                <Text color="fg.muted" flexShrink="0" fontSize="2xs" fontVariantNumeric="tabular-nums">
                  {sourceVideo.width} × {sourceVideo.height} · {sourceVideo.fps} fps
                </Text>
              </HStack>
            </Stack>
          ) : (
            <Stack align="center" color="fg.muted" gap="1.5" justify="center" minH="24" px="4">
              {isLoading ? <Spinner size="sm" /> : <FilmIcon aria-hidden="true" size="18" />}
              <Text fontSize="2xs" textAlign="center">
                {disabled && disabledReason
                  ? disabledReason
                  : isLoading
                    ? t('widgets.video.uploadingClip')
                    : t('widgets.video.uploadOrDropClip')}
              </Text>
            </Stack>
          )}
          <DropTargetOverlay isActive={acceptsActiveDrag} isOver={isOver} label={t('widgets.video.dropInitialVideo')} />
        </DropZone>
        {sourceVideo && previewSrc ? (
          <HStack gap="2">
            <TrimBoundThumb
              fps={sourceVideo.fps}
              frame={sourceVideo.startFrame}
              label={t('widgets.video.trimStart')}
              src={previewSrc}
            />
            <TrimBoundThumb
              fps={sourceVideo.fps}
              frame={sourceVideo.endFrame}
              label={t('widgets.video.trimEnd')}
              src={previewSrc}
            />
          </HStack>
        ) : null}
        {sourceVideo ? (
          <Stack gap="2">
            <Field helpText={t('widgets.video.trimHelp')} label={t('widgets.video.trimStart')}>
              <SliderNumberField
                ariaLabel={t('widgets.video.trimStart')}
                disabled={disabled}
                max={maxFrameIndex}
                min={0}
                step={1}
                value={sourceVideo.startFrame}
                onChange={setStartFrame}
              />
            </Field>
            <Field label={t('widgets.video.trimEnd')}>
              <SliderNumberField
                ariaLabel={t('widgets.video.trimEnd')}
                disabled={disabled}
                max={maxFrameIndex}
                min={0}
                step={1}
                value={sourceVideo.endFrame}
                onChange={setEndFrame}
              />
            </Field>
            <HStack justify="end">
              <Button disabled={isLoading} size="xs" variant="ghost" onClick={handleClear}>
                <XIcon aria-hidden="true" size="12" />
                {t('widgets.video.removeClip')}
              </Button>
            </HStack>
          </Stack>
        ) : null}
        {errorMessage ? (
          <Text aria-live="polite" color="fg.error" fontSize="2xs" role="alert" textWrap="pretty">
            {errorMessage}
          </Text>
        ) : null}
        <Input
          ref={fileInputRef}
          accept="video/*"
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
    areSourceClipsEquivalent(previous.sourceVideo, next.sourceVideo)
);
