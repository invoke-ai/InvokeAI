import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi'
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
      <InvMenuItem color="error.300" icon={<PiTrashSimpleBold />} onClick={handleDelete}>
        {t('boards.deleteBoard')}
      </InvMenuItem>
    </>
  );
};

export default memo(GalleryBoardContextMenuItems);
