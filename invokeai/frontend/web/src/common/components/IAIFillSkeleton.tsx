import { Box, Skeleton } from '@chakra-ui/react';
import { memo } from 'react';

const IAIFillSkeleton = () => {
  return (
    <Skeleton
      sx={{
        position: 'relative',
        height: 'full',
        width: 'full',
        '::before': {
          content: "''",
          display: 'block',
          pt: '100%',
        },
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineStart: 0,
          height: 'full',
          width: 'full',
        }}
      />
    </Skeleton>
  );
};

export default memo(IAIFillSkeleton);
