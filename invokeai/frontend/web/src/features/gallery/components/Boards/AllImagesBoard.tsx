import { MoveBoardDropData } from 'app/components/ImageDnd/typesafeDnd';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { FaImages } from 'react-icons/fa';
import { useDispatch } from 'react-redux';
import GenericBoard from './GenericBoard';

const AllImagesBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();

  const handleAllImagesBoardClick = () => {
    dispatch(boardIdSelected());
  };

  const droppableData: MoveBoardDropData = {
    id: 'all-images-board',
    actionType: 'MOVE_BOARD',
    context: { boardId: null },
  };

  return (
    <GenericBoard
      droppableData={droppableData}
      onClick={handleAllImagesBoardClick}
      isSelected={isSelected}
      icon={FaImages}
      label="All Images"
    />
  );
};

export default AllImagesBoard;
