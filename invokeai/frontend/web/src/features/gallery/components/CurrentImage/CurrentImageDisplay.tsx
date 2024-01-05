import { Flex } from '@chakra-ui/react';
import { memo } from 'react';

import CurrentImageButtons from './CurrentImageButtons';

const CurrentImageDisplay = () => {
  return (
    <Flex
      position="relative"
      flexDirection="column"
      height="100%"
      width="100%"
      rowGap={4}
      alignItems="center"
      justifyContent="center"
    >
      <CurrentImageButtons />
      {/* <CurrentImagePreview /> */}
    </Flex>
  );
};

export default memo(CurrentImageDisplay);
