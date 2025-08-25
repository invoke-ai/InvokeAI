import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import VideoMetadataViewer from 'features/gallery/components/ImageMetadataViewer/VideoMetadataViewer';
import NextPrevItemButtons from 'features/gallery/components/NextPrevItemButtons';
import { selectShouldShowItemDetails } from 'features/ui/store/uiSelectors';
import { VideoViewer } from 'features/video/components/VideoViewer';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useRef, useState } from 'react';
import type { VideoDTO } from 'services/api/types';

import { NoContentForViewer } from './NoContentForViewer';

export const CurrentVideoPreview = memo(({ videoDTO }: { videoDTO: VideoDTO | null }) => {
  const shouldShowItemDetails = useAppSelector(selectShouldShowItemDetails);

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
      {videoDTO && videoDTO.video_url && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center">
          <VideoViewer />
        </Flex>
      )}
      {!videoDTO && <NoContentForViewer />}
      {shouldShowItemDetails && videoDTO && (
        <Box position="absolute" opacity={0.8} top={0} width="full" height="full" borderRadius="base">
          <VideoMetadataViewer video={videoDTO} />
        </Box>
      )}
      <AnimatePresence>
        {shouldShowNextPrevButtons && videoDTO && (
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
CurrentVideoPreview.displayName = 'CurrentVideoPreview';

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
