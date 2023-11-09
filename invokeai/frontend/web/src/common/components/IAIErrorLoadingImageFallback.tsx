import { Box, Flex, Icon } from '@chakra-ui/react';
import { memo } from 'react';
import { FaExclamation } from 'react-icons/fa';

const IAIErrorLoadingImageFallback = () => {
  return (
    <Box
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
      <Flex
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineStart: 0,
          height: 'full',
          width: 'full',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 'base',
          bg: 'base.100',
          color: 'base.500',
          _dark: {
            color: 'base.700',
            bg: 'base.850',
          },
        }}
      >
        <Icon as={FaExclamation} boxSize={16} opacity={0.7} />
      </Flex>
    </Box>
  );
};

export default memo(IAIErrorLoadingImageFallback);
