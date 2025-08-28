import { Box, Flex, Text } from '@invoke-ai/ui-library';
import { useFocusRegion } from 'common/hooks/focus';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import ReactPlayer from 'react-player';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectGeneratedVideo } from 'features/parameters/store/videoSlice';


export const VideoPlayerPanel = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const generatedVideo = useAppSelector(selectGeneratedVideo);

  useFocusRegion('video', ref);


  return (
    <Flex ref={ref} w="full" h="full" flexDirection="column" gap={4}>

      {generatedVideo &&
        <>
          <Box flex={0.75} position="relative" >
            {/* <ReactPlayer
              src={generatedVideo.url}
              width="75%"
              height="75%"
              controls={true}
              style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', maxWidth: '900px' }}
            /> */}
          </Box>
      
      
        </>}
      {!generatedVideo && <Text>No video generated</Text>}

    </Flex>
  );
});

VideoPlayerPanel.displayName = 'VideoPlayerPanel'; 