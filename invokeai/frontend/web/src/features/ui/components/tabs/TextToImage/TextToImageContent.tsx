import { Box, Flex } from '@chakra-ui/react';
import CurrentImageDisplay from 'features/gallery/components/CurrentImageDisplay';

const TextToImageContent = () => {
  return (
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
  );
};

export default TextToImageContent;
