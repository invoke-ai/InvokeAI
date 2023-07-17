import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import { useAppSelector } from 'app/store/storeHooks';
import { selectListAllImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { FaImages } from 'react-icons/fa';
import { useDispatch } from 'react-redux';
import { useListImagesQuery } from 'services/api/endpoints/images';
import GenericBoard from './GenericBoard';

const AllImagesBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();
  const queryArgs = useAppSelector(selectListAllImagesBaseQueryArgs);

  const handleAllImagesBoardClick = () => {
    dispatch(boardIdSelected('all'));
  };

  const { total } = useListImagesQuery(queryArgs, {
    selectFromResult: ({ data }) => ({ total: data?.total ?? 0 }),
  });

  const droppableData: MoveBoardDropData = {
    id: 'all-images-board',
    actionType: 'MOVE_BOARD',
    context: { boardId: 'all' },
  };

  return (
    <GenericBoard
      droppableData={droppableData}
      onClick={handleAllImagesBoardClick}
      isSelected={isSelected}
      icon={FaImages}
      label="All Images"
      badgeCount={total}
    />
  );
};

export default AllImagesBoard;
