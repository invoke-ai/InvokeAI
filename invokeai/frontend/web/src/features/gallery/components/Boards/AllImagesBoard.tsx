import { Flex, Icon, Text } from '@chakra-ui/react';
import { FaImages } from 'react-icons/fa';
import { boardIdSelected } from '../../store/boardSlice';
import { useDispatch } from 'react-redux';
import { IAINoImageFallback } from 'common/components/IAIImageFallback';
import { AnimatePresence } from 'framer-motion';
import { SelectedItemOverlay } from '../SelectedItemOverlay';

const AllImagesBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();

  const handleAllImagesBoardClick = () => {
    dispatch(boardIdSelected());
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
        borderRadius: 'base',
      }}
      onClick={handleAllImagesBoardClick}
    >
      <Flex
        sx={{
          position: 'relative',
          justifyContent: 'center',
          alignItems: 'center',
          borderRadius: 'base',
          w: 'full',
          aspectRatio: '1/1',
        }}
      >
        <IAINoImageFallback iconProps={{ boxSize: 8 }} as={FaImages} />
        <AnimatePresence>
          {isSelected && <SelectedItemOverlay />}
        </AnimatePresence>
      </Flex>
      <Text
        sx={{
          color: isSelected ? 'base.50' : 'base.200',
          fontWeight: isSelected ? 600 : undefined,
          fontSize: 'xs',
        }}
      >
        All Images
      </Text>
    </Flex>
  );
};

export default AllImagesBoard;
