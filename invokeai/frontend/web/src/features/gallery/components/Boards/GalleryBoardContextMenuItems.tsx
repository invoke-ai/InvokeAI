import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import type { BoardDTO } from 'services/api/types';

type Props = {
  board: BoardDTO;
  setBoardToDelete?: (board?: BoardDTO) => void;
};

const GalleryBoardContextMenuItems = ({ board, setBoardToDelete }: Props) => {
  const { t } = useTranslation();
  const handleDelete = useCallback(() => {
    if (!setBoardToDelete) {
      return;
    }
    setBoardToDelete(board);
  }, [board, setBoardToDelete]);

  return (
    <>
      <InvMenuItem color="error.300" icon={<FaTrash />} onClick={handleDelete}>
        {t('boards.deleteBoard')}
      </InvMenuItem>
    </>
  );
};

export default memo(GalleryBoardContextMenuItems);
