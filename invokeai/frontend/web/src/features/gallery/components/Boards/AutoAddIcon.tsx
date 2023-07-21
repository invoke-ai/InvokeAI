import { Flex, Icon } from '@chakra-ui/react';
import { FaPlus } from 'react-icons/fa';

const AutoAddIcon = () => {
  return (
    <Flex
      sx={{
        position: 'absolute',
        insetInlineStart: 0,
        top: 0,
        p: 1,
      }}
    >
      <Icon as={FaPlus} sx={{ fill: 'accent.500' }} />
    </Flex>
  );
};

export default AutoAddIcon;
