import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Box, Button, Flex, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useClipboard } from 'common/hooks/useClipboard';
import { useDownloadItem } from 'common/hooks/useDownloadImage';
import { useMediaUrl } from 'features/auth/store/mediaCookieRefresh';
import { useDeleteVideoModalApi } from 'features/deleteVideoModal/store/state';
import { multipleVideoDndSource, singleVideoDndSource } from 'features/dnd/dnd';
import { firefoxDndFix } from 'features/dnd/util';
import VideoMetadataViewer from 'features/gallery/components/ImageMetadataViewer/VideoMetadataViewer';
import NextPrevItemButtons from 'features/gallery/components/NextPrevItemButtons';
import { useNextPrevItemNavigation } from 'features/gallery/components/useNextPrevItemNavigation';
import { selectSelectedBoardId, selectSelection } from 'features/gallery/store/gallerySelectors';
import { isVideoName } from 'features/gallery/store/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import {
  selectActiveTab,
  selectShouldShowItemDetails,
  selectShouldShowProgressInViewer,
} from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold, PiCopyBold, PiDownloadSimpleBold, PiTrashSimpleBold, PiXBold } from 'react-icons/pi';
import type { VideoDTO } from 'services/api/types';

import { useImageViewerContext } from './context';
import { NoContentForViewer } from './NoContentForViewer';
import { ProgressImage } from './ProgressImage2';
import { ProgressIndicator } from './ProgressIndicator2';
import { VideoPlayButtonOverlay } from './VideoPlayButtonOverlay';

type Props = {
  videoDTO: VideoDTO | null;
};

/**
 * Counterpart to CurrentImagePreview for videos. A single <video> element spans both states:
 *
 *  - **idle**: muted, no controls. Without a `poster` attribute the browser decodes and
 *    displays the video's actual first frame at full resolution (much sharper than the
 *    small WebP gallery thumbnail upscaled to fit the viewer). A centered play button
 *    overlay sits on top.
 *  - **playing**: native HTML5 controls + audio. The element is the same DOM node, so the
 *    decoded buffer carries over — no reload when the user hits play.
 *
 * Changing the selected video swaps the element via `key={videoName}`, which discards the
 * old playback state cleanly.
 *
 * Mirrors CurrentImagePreview's progress overlay so denoise previews from a new render
 * appear on top of the previously-loaded video. Without this, a freshly generated render's
 * progress images had nowhere to display whenever a video was the last-selected gallery
 * item (and the user only saw the static first-frame still until the new video finished).
 */
export const CurrentVideoPreview = memo(({ videoDTO }: Props) => {
  const videoUrl = useMediaUrl(videoDTO?.video_url);
  const { t } = useTranslation();
  const store = useAppStore();
  const videoName = videoDTO?.video_name ?? null;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const shouldShowItemDetails = useAppSelector(selectShouldShowItemDetails);
  const activeTab = useAppSelector(selectActiveTab);
  const deleteVideoModal = useDeleteVideoModalApi();
  const { downloadItem } = useDownloadItem();
  const clipboard = useClipboard();
  const { $progressEvent, $progressImage, onLoadImage } = useImageViewerContext();
  const progressEvent = useStore($progressEvent);
  const progressImage = useStore($progressImage);
  const withProgress = shouldShowProgressInViewer && progressImage !== null;
  const { goToPreviousImage, goToNextImage, isFetching } = useNextPrevItemNavigation();

  // Whenever the selected video changes, drop back to the idle still + play overlay.
  useEffect(() => {
    setIsPlaying(false);
  }, [videoName]);

  // Register the viewer's <video> as a drag source so users can drag the currently-displayed
  // video onto node fields (e.g. a Video Primitive's "Starting Video" input) directly from
  // the viewer, just like they can from the gallery thumbnail. Mirrors GalleryVideoItem's
  // setup. Without this, the bare <video> element has no drag handler and the drop target
  // sees nothing it can accept.
  useEffect(() => {
    const element = videoRef.current;
    if (!element || !videoDTO) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData: () => {
          // Honor any active gallery multi-selection so dropping onto a board moves the whole
          // batch, matching the gallery thumbnail's behavior.
          const state = store.getState();
          const selection = selectSelection(state);
          const boardId = selectSelectedBoardId(state);
          if (selection.length > 1 && selection.includes(videoDTO.video_name)) {
            const video_names = selection.filter(isVideoName);
            const image_names = selection.filter((n) => !isVideoName(n));
            return multipleVideoDndSource.getData({
              video_names,
              image_names,
              board_id: boardId,
            });
          }
          return singleVideoDndSource.getData({ videoDTO }, videoDTO.video_name);
        },
      })
    );
  }, [videoDTO, store]);

  const handlePlay = useCallback(() => {
    setIsPlaying(true);
    // The ref points at the same element we'll re-render with controls/audio; calling
    // play() here keeps the user gesture wired to playback without waiting for React.
    void videoRef.current?.play();
  }, []);

  // Close: stop playback and drop back to the first-frame preview + play overlay. We
  // explicitly pause() because toggling React's `controls` prop hides the chrome but does
  // not stop playback. Seeking back to ~0 nudges the decoder to re-paint the first frame
  // (mirroring handleLoadedMetadata's near-zero seek trick).
  const handleClose = useCallback(() => {
    const el = videoRef.current;
    if (el) {
      el.pause();
      try {
        el.currentTime = 0.0001;
      } catch {
        // Some browsers throw if metadata isn't fully ready yet; harmless.
      }
    }
    setIsPlaying(false);
  }, []);

  const handleDelete = useCallback(async () => {
    if (!videoDTO) {
      return;
    }
    try {
      await deleteVideoModal.delete([videoDTO.video_name]);
    } catch {
      // user canceled the confirmation dialog
    }
  }, [deleteVideoModal, videoDTO]);

  const handleDownload = useCallback(() => {
    if (!videoDTO) {
      return;
    }
    void downloadItem(videoDTO.video_url, videoDTO.video_name);
  }, [downloadItem, videoDTO]);

  const handleOpenInNewTab = useCallback(() => {
    if (!videoDTO) {
      return;
    }
    window.open(videoDTO.video_url, '_blank', 'noopener,noreferrer');
  }, [videoDTO]);

  // Cross-browser clipboard support for raw `video/*` MIME types doesn't really exist — Chrome
  // and Firefox both reject anything outside a small allow-list (image/png, image/jpeg, text).
  // So instead we grab the currently-displayed frame off the <video> element via a canvas and
  // hand the resulting PNG to the standard image-clipboard path. The video is same-origin so
  // the canvas doesn't taint.
  const handleCopyFrame = useCallback(async () => {
    const el = videoRef.current;
    if (!el || !el.videoWidth || !el.videoHeight) {
      return;
    }
    try {
      const canvas = document.createElement('canvas');
      canvas.width = el.videoWidth;
      canvas.height = el.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Unable to acquire 2D canvas context');
      }
      ctx.drawImage(el, 0, 0);
      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob((b) => resolve(b), 'image/png');
      });
      if (!blob) {
        throw new Error('Unable to encode frame as PNG');
      }
      clipboard.writeImage(blob, () => {
        toast({
          id: 'IMAGE_COPIED',
          title: t('toast.imageCopied'),
          status: 'success',
        });
      });
    } catch (err) {
      toast({
        id: 'PROBLEM_COPYING_IMAGE',
        title: t('toast.problemCopyingImage'),
        description: String(err),
        status: 'error',
      });
    }
  }, [clipboard, t]);

  // Mirror CurrentImagePreview's hover-driven next/prev gating so the arrows only intrude
  // while the user is interacting with the viewer.
  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] = useState<boolean>(false);
  const timeoutId = useRef(0);
  const onMouseOver = useCallback(() => {
    setShouldShowNextPrevButtons(true);
    window.clearTimeout(timeoutId.current);
  }, []);
  const onMouseOut = useCallback(() => {
    timeoutId.current = window.setTimeout(() => {
      setShouldShowNextPrevButtons(false);
    }, 500);
  }, []);

  const handleViewerArrowNavigation = useCallback(
    (event: KeyboardEvent, navigate: () => void) => {
      if (!navigationApi.isViewerArrowNavigationMode(activeTab) || !videoDTO || isFetching) {
        return;
      }
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      event.preventDefault();
      navigate();
    },
    [activeTab, videoDTO, isFetching]
  );

  const onHotkeyPrevImage = useCallback(
    (event: KeyboardEvent) => {
      handleViewerArrowNavigation(event, goToPreviousImage);
    },
    [goToPreviousImage, handleViewerArrowNavigation]
  );

  const onHotkeyNextImage = useCallback(
    (event: KeyboardEvent) => {
      handleViewerArrowNavigation(event, goToNextImage);
    },
    [goToNextImage, handleViewerArrowNavigation]
  );

  useRegisteredHotkeys({
    id: 'galleryNavLeft',
    category: 'gallery',
    callback: onHotkeyPrevImage,
    options: { preventDefault: true },
    dependencies: [onHotkeyPrevImage],
  });

  useRegisteredHotkeys({
    id: 'galleryNavRight',
    category: 'gallery',
    callback: onHotkeyNextImage,
    options: { preventDefault: true },
    dependencies: [onHotkeyNextImage],
  });

  // Analogous to <DndImage onLoad={onLoadImage}> in the image viewer: clear any stale
  // denoise progress overlay once the new video's metadata is in. Without this, the
  // ImageViewerContext atom stays set after a video render (there's no image load to
  // trigger its clear), so the overlay sticks over the freshly-selected video forever.
  //
  // Also force a first-frame paint via a near-zero seek. With preload="metadata" some
  // browsers populate dimensions/duration but don't actually decode and display the first
  // video frame until playback or a seek — the element just shows its black background.
  // Setting currentTime to 0.0001 nudges the decoder to paint without measurably advancing.
  const handleLoadedMetadata = useCallback(() => {
    onLoadImage();
    const el = videoRef.current;
    if (el && !isPlaying && el.currentTime === 0) {
      try {
        el.currentTime = 0.0001;
      } catch {
        // Some browsers throw if metadata isn't fully ready yet; harmless.
      }
    }
  }, [isPlaying, onLoadImage]);

  if (!videoDTO) {
    return <NoContentForViewer />;
  }

  return (
    <Flex
      onMouseOver={onMouseOver}
      onMouseOut={onMouseOut}
      width="full"
      height="full"
      alignItems="center"
      justifyContent="center"
      position="relative"
    >
      <video
        key={videoName ?? undefined}
        ref={videoRef}
        // Resolves to /api/v1/videos/i/{name}/full, which supports HTTP Range — used both
        // for first-frame decode (preload=metadata) and for scrub during playback.
        src={videoUrl}
        preload="metadata"
        muted={!isPlaying}
        playsInline
        controls={isPlaying}
        onLoadedMetadata={handleLoadedMetadata}
        style={{
          maxWidth: '100%',
          maxHeight: '100%',
          borderRadius: 4,
          outline: 'none',
          background: 'black',
        }}
      />
      {!isPlaying && !withProgress && <VideoPlayButtonOverlay onClick={handlePlay} />}
      {shouldShowItemDetails && !withProgress && (
        <Box position="absolute" opacity={0.8} top={0} width="full" height="full" borderRadius="base">
          <VideoMetadataViewer video={videoDTO} />
        </Box>
      )}
      {withProgress && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center" bg="base.900">
          <ProgressImage progressImage={progressImage} />
          {progressEvent && (
            <ProgressIndicator progressEvent={progressEvent} position="absolute" top={6} right={6} size={8} />
          )}
        </Flex>
      )}
      {/* Top action bar, right-aligned. Auto-sized Flex anchored to insetInlineEnd leaves the
          rest of the viewer click-through so clicking the video still pauses native playback.
          Order: open in new tab, copy frame, download, delete, then the labelled close button
          farthest right (only while the player is active). */}
      <Flex position="absolute" top={2} insetInlineEnd={2} zIndex={2} gap={1}>
        <IconButton
          aria-label={t('common.openInNewTab')}
          tooltip={t('common.openInNewTab')}
          icon={<PiArrowSquareOutBold />}
          onClick={handleOpenInNewTab}
          variant="solid"
          size="sm"
        />
        <IconButton
          aria-label={t('gallery.copyVideoFrame')}
          tooltip={t('gallery.copyVideoFrame')}
          icon={<PiCopyBold />}
          onClick={handleCopyFrame}
          variant="solid"
          size="sm"
        />
        <IconButton
          aria-label={t('gallery.download')}
          tooltip={t('gallery.download')}
          icon={<PiDownloadSimpleBold />}
          onClick={handleDownload}
          variant="solid"
          size="sm"
        />
        <IconButton
          aria-label={t('gallery.deleteVideo', { count: 1 })}
          tooltip={t('gallery.deleteVideo', { count: 1 })}
          icon={<PiTrashSimpleBold />}
          onClick={handleDelete}
          colorScheme="error"
          variant="solid"
          size="sm"
        />
        {isPlaying && (
          <Button leftIcon={<PiXBold />} onClick={handleClose} variant="solid" size="sm">
            {t('gallery.closeVideoPlayer')}
          </Button>
        )}
      </Flex>
      <AnimatePresence>
        {shouldShowNextPrevButtons && (
          <Box
            as={motion.div}
            key="nextPrevButtons"
            initial={initial}
            animate={animateArrows}
            exit={exit}
            position="absolute"
            top={0}
            right={0}
            bottom={0}
            left={0}
            pointerEvents="none"
          >
            <NextPrevItemButtons />
          </Box>
        )}
      </AnimatePresence>
    </Flex>
  );
});

const initial: AnimationProps['initial'] = {
  opacity: 0,
};
const animateArrows: AnimationProps['animate'] = {
  opacity: 1,
  transition: { duration: 0.07 },
};
const exit: AnimationProps['exit'] = {
  opacity: 0,
  transition: { duration: 0.07 },
};

CurrentVideoPreview.displayName = 'CurrentVideoPreview';
