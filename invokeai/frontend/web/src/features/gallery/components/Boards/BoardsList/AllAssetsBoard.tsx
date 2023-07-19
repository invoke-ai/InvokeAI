import {
  ASSETS_CATEGORIES,
  INITIAL_IMAGE_LIMIT,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import { FaFileImage } from 'react-icons/fa';
import { useDispatch } from 'react-redux';
import {
  ListImagesArgs,
  useListImagesQuery,
} from 'services/api/endpoints/images';
import GenericBoard from './GenericBoard';

const baseQueryArg: ListImagesArgs = {
  categories: ASSETS_CATEGORIES,
  offset: 0,
  limit: INITIAL_IMAGE_LIMIT,
  is_intermediate: false,
};

const AllAssetsBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();

  const handleClick = () => {
    dispatch(boardIdSelected('assets'));
  };

  const { total } = useListImagesQuery(baseQueryArg, {
    selectFromResult: ({ data }) => ({ total: data?.total ?? 0 }),
  });

  // TODO: Do we support making 'images' 'assets? if yes, we need to handle this
  // const droppableData: MoveBoardDropData = {
  //   id: 'all-images-board',
  //   actionType: 'MOVE_BOARD',
  //   context: { boardId: 'assets' },
  // };

  return (
    <GenericBoard
      onClick={handleClick}
      isSelected={isSelected}
      icon={FaFileImage}
      label="All Assets"
      badgeCount={total}
    />
  );
};

export default AllAssetsBoard;
