import { Flex, Icon } from '@chakra-ui/react';
import { memo } from 'react';
import { FaImage } from 'react-icons/fa';

const SelectImagePlaceholder = () => {
  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        // bg: 'base.800',
        borderRadius: 'base',
        alignItems: 'center',
        justifyContent: 'center',
        aspectRatio: '1/1',
      }}
    >
      <Icon color="base.400" boxSize={32} as={FaImage}></Icon>
    </Flex>
  );
};

export default memo(SelectImagePlaceholder);
