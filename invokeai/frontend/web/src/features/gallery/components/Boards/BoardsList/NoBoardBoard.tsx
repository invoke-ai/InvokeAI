import { Text } from '@chakra-ui/react';
import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import {
  INITIAL_IMAGE_LIMIT,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import { FaFolderOpen } from 'react-icons/fa';
import { useDispatch } from 'react-redux';
import {
  ListImagesArgs,
  useListImagesQuery,
} from 'services/api/endpoints/images';
import GenericBoard from './GenericBoard';

const baseQueryArg: ListImagesArgs = {
  board_id: 'none',
  offset: 0,
  limit: INITIAL_IMAGE_LIMIT,
  is_intermediate: false,
};

const NoBoardBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();

  const handleClick = () => {
    dispatch(boardIdSelected('no_board'));
  };

  const { total } = useListImagesQuery(baseQueryArg, {
    selectFromResult: ({ data }) => ({ total: data?.total ?? 0 }),
  });

  // TODO: Do we support making 'images' 'assets? if yes, we need to handle this
  const droppableData: MoveBoardDropData = {
    id: 'all-images-board',
    actionType: 'MOVE_BOARD',
    context: { boardId: 'no_board' },
  };

  return (
    <GenericBoard
      droppableData={droppableData}
      dropLabel={<Text fontSize="md">Move</Text>}
      onClick={handleClick}
      isSelected={isSelected}
      icon={FaFolderOpen}
      label="No Board"
      badgeCount={total}
    />
  );
};

export default NoBoardBoard;
