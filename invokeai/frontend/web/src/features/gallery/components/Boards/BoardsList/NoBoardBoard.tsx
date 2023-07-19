import {
  INITIAL_IMAGE_LIMIT,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import { FaFolder } from 'react-icons/fa';
import { useDispatch } from 'react-redux';
import {
  ListImagesArgs,
  useListImagesQuery,
} from 'services/api/endpoints/images';
import GenericBoard from './GenericBoard';
import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';

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
      onClick={handleClick}
      isSelected={isSelected}
      icon={FaFolder}
      label="No Board"
      badgeCount={total}
    />
  );
};

export default NoBoardBoard;
