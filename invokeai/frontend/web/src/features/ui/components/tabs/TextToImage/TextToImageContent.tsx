import { Box, Flex } from '@chakra-ui/react';
import CurrentImageDisplay from 'features/gallery/components/CurrentImageDisplay';
import MediaQuery from 'react-responsive';

const TextToImageContent = () => {
  return (
    <>
      <MediaQuery minDeviceWidth={768}>
        <Box
          sx={{
            width: '100%',
            height: '100%',
            borderRadius: 'base',
            bg: 'base.850',
          }}
        >
          <Flex sx={{ p: 4, width: '100%', height: '100%' }}>
            <CurrentImageDisplay />
          </Flex>
        </Box>
      </MediaQuery>
      <MediaQuery maxDeviceWidth={768}>
        <Box
          sx={{
            position: 'relative',
            width: 'full',
            height: 'full',
            bg: 'base.850',
          }}
        >
          <Flex sx={{ height: '100%' }}>
            <CurrentImageDisplay />
          </Flex>
        </Box>
      </MediaQuery>
    </>
  );
};

export default TextToImageContent;
