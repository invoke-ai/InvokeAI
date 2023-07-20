import { BoardId } from 'features/gallery/store/gallerySlice';
import { memo } from 'react';

type Props = {
  board_id: BoardId;
};

const SystemBoardContextMenuItems = ({ board_id }: Props) => {
  return <></>;
};

export default memo(SystemBoardContextMenuItems);
