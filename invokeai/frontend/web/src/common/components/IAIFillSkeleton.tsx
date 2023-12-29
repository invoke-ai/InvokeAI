import type { SystemStyleObject } from '@chakra-ui/react';
import { Box, Skeleton } from '@chakra-ui/react';
import { memo } from 'react';

const skeletonStyles: SystemStyleObject = {
  position: 'relative',
  height: 'full',
  width: 'full',
  '::before': {
    content: "''",
    display: 'block',
    pt: '100%',
  },
};

const IAIFillSkeleton = () => {
  return (
    <Skeleton sx={skeletonStyles}>
      <Box
        position="absolute"
        top={0}
        insetInlineStart={0}
        height="full"
        width="full"
      />
    </Skeleton>
  );
};

export default memo(IAIFillSkeleton);
