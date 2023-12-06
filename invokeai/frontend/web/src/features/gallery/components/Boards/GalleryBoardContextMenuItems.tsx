import { MenuItem } from '@chakra-ui/react';
import { memo, useCallback, useMemo } from 'react';
import { FaTrash } from 'react-icons/fa';
import { BoardDTO } from 'services/api/types';
import { useTranslation } from 'react-i18next';
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

  const isDeleteDisabled = useMemo(() => {
    if (board?.actions) {
      return board.actions.delete === false;
    } else {
      return false;
    }
  }, [board]);

  return (
    <>
      <MenuItem
        sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
        icon={<FaTrash />}
        onClick={handleDelete}
        isDisabled={isDeleteDisabled}
      >
        {t('boards.deleteBoard')}
      </MenuItem>
    </>
  );
};

export default memo(GalleryBoardContextMenuItems);
