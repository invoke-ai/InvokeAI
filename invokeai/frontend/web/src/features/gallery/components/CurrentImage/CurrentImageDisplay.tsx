import { Flex } from '@chakra-ui/react';

import CurrentImageButtons from './CurrentImageButtons';
import CurrentImagePreview from './CurrentImagePreview';
import { memo } from 'react';

const CurrentImageDisplay = () => {
  return (
    <Flex
      sx={{
        position: 'relative',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        rowGap: 4,
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <CurrentImageButtons />
      <CurrentImagePreview />
    </Flex>
  );
};

export default memo(CurrentImageDisplay);
