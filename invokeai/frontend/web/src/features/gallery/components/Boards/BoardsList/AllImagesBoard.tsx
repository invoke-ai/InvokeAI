import {
  IMAGE_CATEGORIES,
  INITIAL_IMAGE_LIMIT,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import { FaImages } from 'react-icons/fa';
import { useDispatch } from 'react-redux';
import {
  ListImagesArgs,
  useListImagesQuery,
} from 'services/api/endpoints/images';
import GenericBoard from './GenericBoard';

const baseQueryArg: ListImagesArgs = {
  categories: IMAGE_CATEGORIES,
  offset: 0,
  limit: INITIAL_IMAGE_LIMIT,
  is_intermediate: false,
};

const AllImagesBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();

  const handleClick = () => {
    dispatch(boardIdSelected('images'));
  };

  const { total } = useListImagesQuery(baseQueryArg, {
    selectFromResult: ({ data }) => ({ total: data?.total ?? 0 }),
  });

  // TODO: Do we support making 'images' 'assets? if yes, we need to handle this
  // const droppableData: MoveBoardDropData = {
  //   id: 'all-images-board',
  //   actionType: 'MOVE_BOARD',
  //   context: { boardId: 'images' },
  // };

  return (
    <GenericBoard
      onClick={handleClick}
      isSelected={isSelected}
      icon={FaImages}
      label="All Images"
      badgeCount={total}
    />
  );
};

export default AllImagesBoard;
