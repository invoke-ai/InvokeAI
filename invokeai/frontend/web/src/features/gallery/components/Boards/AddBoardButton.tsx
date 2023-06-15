import { Flex, Icon, Text } from '@chakra-ui/react';
import { FaPlus } from 'react-icons/fa';

const AddBoardButton = () => {
  return (
    <Flex
      sx={{
        flexDir: 'column',
        justifyContent: 'space-between',
        alignItems: 'center',
        cursor: 'pointer',
        w: 'full',
        h: 'full',
        gap: 1,
      }}
    >
      <Flex
        sx={{
          justifyContent: 'center',
          alignItems: 'center',
          borderWidth: '1px',
          borderRadius: 'base',
          borderColor: 'base.800',
          w: 'full',
          h: 'full',
          aspectRatio: '1/1',
        }}
      >
        <Icon boxSize={8} color="base.700" as={FaPlus} />
      </Flex>
      <Text sx={{ color: 'base.200', fontSize: 'xs' }}>New Board</Text>
    </Flex>
  );
};

export default AddBoardButton;
