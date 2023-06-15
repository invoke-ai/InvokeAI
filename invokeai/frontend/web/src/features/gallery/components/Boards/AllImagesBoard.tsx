import { Flex, Icon, Text } from '@chakra-ui/react';
import { FaImages } from 'react-icons/fa';
import { boardIdSelected } from '../../store/boardSlice';
import { useDispatch } from 'react-redux';

const AllImagesBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();

  const handleAllImagesBoardClick = () => {
    dispatch(boardIdSelected(null));
  };

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
      onClick={handleAllImagesBoardClick}
    >
      <Flex
        sx={{
          justifyContent: 'center',
          alignItems: 'center',
          borderWidth: '1px',
          borderRadius: 'base',
          borderColor: isSelected ? 'base.500' : 'base.800',
          w: 'full',
          h: 'full',
          aspectRatio: '1/1',
        }}
      >
        <Icon boxSize={8} color="base.700" as={FaImages} />
      </Flex>
      <Text sx={{ color: 'base.200', fontSize: 'xs' }}>All Images</Text>
    </Flex>
  );
};

export default AllImagesBoard;
