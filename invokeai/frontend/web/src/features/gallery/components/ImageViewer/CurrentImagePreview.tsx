import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useMediaUrl } from 'features/auth/store/mediaCookieRefresh';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { DndImage } from 'features/dnd/DndImage';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevItemButtons from 'features/gallery/components/NextPrevItemButtons';
import { useNextPrevItemNavigation } from 'features/gallery/components/useNextPrevItemNavigation';
import { $gallerySelection } from 'features/gallery/store/gallerySelectionSource';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import {
  selectActiveTab,
  selectShouldShowItemDetails,
  selectShouldShowProgressInViewer,
} from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import type { ImageDTO } from 'services/api/types';

import { useImageViewerContext } from './context';
import { NoContentForViewer } from './NoContentForViewer';
import { ProgressImage } from './ProgressImage2';
import { ProgressImageTiles } from './ProgressImageTiles';
import { ProgressIndicator } from './ProgressIndicator2';

export const CurrentImagePreview = memo(({ imageDTO }: { imageDTO: ImageDTO | null }) => {
  const activeTab = useAppSelector(selectActiveTab);
  const selectedImageName = useAppSelector(selectLastSelectedItem);
  const shouldShowItemDetails = useAppSelector(selectShouldShowItemDetails);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const { goToPreviousImage, goToNextImage, isFetching } = useNextPrevItemNavigation();
  const {
    onLoadImage,
    $progressEvent,
    $progressImage,
    $activeProgressData,
    $isProgressImageResolving,
    $isTemporarilyShowingSelectedImage,
    revealMachine,
  } = useImageViewerContext();
  const progressEvent = useStore($progressEvent);
  const progressImage = useStore($progressImage);
  const activeProgressData = useStore($activeProgressData);
  const isProgressImageResolving = useStore($isProgressImageResolving);
  const isTemporarilyShowingSelectedImage = useStore($isTemporarilyShowingSelectedImage);
  const selection = useStore($gallerySelection);
  const [imageToRender, setImageToRender] = useState<ImageDTO | null>(null);

  // The reveal gate below deliberately preloads the *thumbnail*, not the full-resolution image. The
  // progress overlay covers this element until onLoadImage fires, so gating on the multi-megabyte
  // `/full` response would hold a stale latent preview on screen for that entire download on a slow
  // connection. The 256px thumbnail is roughly 100x smaller and is typically higher resolution than
  // the preview it replaces; DndImage renders it via Chakra's `fallbackSrc` and swaps the full image
  // in, in place, once that finishes loading.
  //
  // The URL must go through useMediaUrl so it is byte-identical to the one DndImage requests. The
  // media cookie version is a query parameter, so a mismatch is a different key and the bytes are
  // fetched twice (measured: 2 requests mismatched vs 1 matched). Note the reuse here is the
  // document's list of available images, which is keyed by URL and is not the HTTP cache — it still
  // holds in multiuser mode, where images are served `Cache-Control: private, no-store`.
  const previewSrc = useMediaUrl(imageDTO?.thumbnail_url);

  useEffect(() => {
    if (!selectedImageName) {
      setImageToRender(null);
      return;
    }

    if (!imageDTO || imageToRender?.image_name === imageDTO.image_name) {
      return;
    }

    let canceled = false;

    const onReady = () => {
      if (canceled) {
        return;
      }
      setImageToRender(imageDTO);
      // Resolve the progress overlay as soon as the thumbnail settles — on success *or* error.
      // Relying on DndImage's onLoad alone leaves the overlay stuck whenever the image fails to
      // load, because Chakra reports that as onError instead. The session id lets the lifecycle
      // attribute the load, so a late-settling thumbnail from an earlier session cannot cut a
      // different session's resolve illusion short.
      onLoadImage(imageDTO.session_id ?? null);
    };

    if (typeof window === 'undefined' || !previewSrc) {
      onReady();
      return;
    }

    const preloader = new window.Image();

    preloader.onload = onReady;
    preloader.onerror = onReady;
    preloader.src = previewSrc;

    if (preloader.complete) {
      onReady();
    }

    return () => {
      canceled = true;
      preloader.onload = null;
      preloader.onerror = null;
    };
  }, [imageDTO, imageToRender?.image_name, onLoadImage, previewSrc, selectedImageName]);

  const hasProgressImage = progressImage !== null;

  // The sequencing lives in the shared machine — see selectedItemReveal.ts. The image path only
  // renders an image once its preload has settled, so whatever is rendered here has painted.
  // Registers this component as a live driver of the machine, so selections landing while the
  // viewer shows neither preview are settled rather than replayed on return.
  useEffect(() => revealMachine.attach(), [revealMachine]);

  useEffect(() => {
    revealMachine.sync({
      selection,
      renderedItemName: imageToRender?.image_name ?? null,
      isMediaReady: imageToRender !== null,
      shouldShowProgressInViewer,
      hasProgressImage,
      isProgressImageResolving,
    });
  }, [hasProgressImage, imageToRender, isProgressImageResolving, revealMachine, selection, shouldShowProgressInViewer]);

  // Show and hide the next/prev buttons on mouse move
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
      if (!navigationApi.isViewerArrowNavigationMode(activeTab) || !imageToRender || isFetching) {
        return;
      }
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      event.preventDefault();
      navigate();
    },
    [activeTab, imageToRender, isFetching]
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

  // The loaded image identifies its session so the viewer can tell a late load from an earlier
  // session apart from the one whose preview is currently retained (see onLoadImage).
  const onLoadRenderedImage = useCallback(() => {
    onLoadImage(imageToRender?.session_id ?? null);
  }, [imageToRender?.session_id, onLoadImage]);

  const withProgress = shouldShowProgressInViewer && hasProgressImage && !isTemporarilyShowingSelectedImage;
  // When more than one session is generating concurrently (multi-GPU), tile their previews instead of
  // showing only the most recent one.
  const withTiledProgress = withProgress && activeProgressData.length > 1;

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
      {imageToRender && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center">
          <DndImage imageDTO={imageToRender} onLoad={onLoadRenderedImage} borderRadius="base" />
        </Flex>
      )}
      {!imageToRender && <NoContentForViewer />}
      {withProgress && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center" bg="base.900">
          {withTiledProgress ? (
            <ProgressImageTiles data={activeProgressData} />
          ) : (
            <>
              <ProgressImage progressImage={progressImage} />
              {progressEvent && (
                <ProgressIndicator progressEvent={progressEvent} position="absolute" top={6} right={6} size={8} />
              )}
            </>
          )}
        </Flex>
      )}
      <Flex flexDir="column" gap={2} position="absolute" top={0} insetInlineStart={0} alignItems="flex-start">
        <CanvasAlertsInvocationProgress />
      </Flex>
      {/* Gated on the reveal state itself, not only on !withProgress (which the reveal turns
          off): the reveal exists to make a mid-render click visibly land, and the full-screen
          metadata panel would drop exactly on top of the just-revealed image for the whole
          window. Mirrors CurrentVideoPreview's gate. */}
      {shouldShowItemDetails && imageToRender && !isTemporarilyShowingSelectedImage && !withProgress && (
        <Box position="absolute" opacity={0.8} top={0} width="full" height="full" borderRadius="base">
          <ImageMetadataViewer image={imageToRender} />
        </Box>
      )}
      <AnimatePresence>
        {shouldShowNextPrevButtons && imageToRender && (
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
CurrentImagePreview.displayName = 'CurrentImagePreview';

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
