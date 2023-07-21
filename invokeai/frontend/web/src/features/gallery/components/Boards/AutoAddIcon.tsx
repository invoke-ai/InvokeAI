import { Badge, Flex, Icon } from '@chakra-ui/react';
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
      <Badge
        variant="solid"
        sx={{ bg: 'accent.400', _dark: { bg: 'accent.500' } }}
      >
        auto
      </Badge>
    </Flex>
  );
};

export default AutoAddIcon;
