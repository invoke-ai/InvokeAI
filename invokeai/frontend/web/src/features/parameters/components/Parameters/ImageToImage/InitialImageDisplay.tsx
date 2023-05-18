import { Flex } from '@chakra-ui/react';
import InitialImagePreview from './InitialImagePreview';
import InitialImageButtons from 'common/components/InitialImageButtons';

const InitialImageDisplay = () => {
  return (
    <Flex
      sx={{
        position: 'relative',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        rowGap: 4,
        borderRadius: 'base',
        alignItems: 'center',
        justifyContent: 'center',
        bg: 'base.850',
        p: 4,
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
        <InitialImageButtons />
        <InitialImagePreview />
      </Flex>
    </Flex>
  );
};

export default InitialImageDisplay;
