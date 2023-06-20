import { Flex, Icon, Spinner, Text } from '@chakra-ui/react';
import { useCallback } from 'react';
import { FaPlus } from 'react-icons/fa';
import { useCreateBoardMutation } from 'services/apiSlice';

const DEFAULT_BOARD_NAME = 'My Board';

const AddBoardButton = () => {
  const [createBoard, { isLoading }] = useCreateBoardMutation();

  const handleCreateBoard = useCallback(() => {
    createBoard(DEFAULT_BOARD_NAME);
  }, [createBoard]);

  return (
    <Flex
      onClick={isLoading ? undefined : handleCreateBoard}
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
        {isLoading ? (
          <Spinner />
        ) : (
          <Icon boxSize={8} color="base.700" as={FaPlus} />
        )}
      </Flex>
      <Text sx={{ color: 'base.200', fontSize: 'xs' }}>New Board</Text>
    </Flex>
  );
};

export default AddBoardButton;
