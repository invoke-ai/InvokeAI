import { Box, Flex } from '@chakra-ui/react';
import CurrentImageDisplay from 'features/gallery/components/CurrentImageDisplay';

const TextToImageTabMain = () => {
  return (
    <Box
      layerStyle={'first'}
      sx={{
        position: 'relative',
        width: '100%',
        height: '100%',
        p: 4,
        borderRadius: 'base',
      }}
    >
      <Flex
        sx={{
          width: '100%',
          height: '100%',
        }}
      >
        <CurrentImageDisplay />
      </Flex>
    </Box>
  );
};

export default TextToImageTabMain;
