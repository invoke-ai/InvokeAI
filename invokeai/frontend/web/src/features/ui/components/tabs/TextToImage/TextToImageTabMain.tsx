import { Box, Flex } from '@chakra-ui/react';
import CurrentImageDisplay from 'features/gallery/components/CurrentImageDisplay';

const TextToImageTabMain = () => {
  return (
    <Box
      sx={{
        position: 'relative',
        width: '100%',
        height: '100%',
        borderRadius: 'base',
        bg: 'base.850',
        p: 4,
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
