import { Flex } from '@chakra-ui/react';
import InitialImagePreview from './InitialImagePreview';

const InitialImageDisplay = () => {
  return (
    <Flex
      layerStyle={'first'}
      sx={{
        position: 'relative',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        rowGap: 4,
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
        borderRadius: 'base',
      }}
    >
      <Flex
        flexDirection="column"
        sx={{
          w: 'full',
          h: 'full',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 4,
        }}
      >
        <InitialImagePreview />
      </Flex>
    </Flex>
  );
};

export default InitialImageDisplay;
