import { Icon } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

type Props = {
  privateBoard: boolean;
};

const AddBoardButton = ({ privateBoard }: Props) => {
  const { t } = useTranslation();
  const [createBoard] = useCreateBoardMutation();
  const DEFAULT_BOARD_NAME = t('boards.myBoard');
  const handleCreateBoard = useCallback(() => {
    createBoard({ DEFAULT_BOARD_NAME, privateBoard });
  }, [createBoard, DEFAULT_BOARD_NAME, privateBoard]);

  return (
    <Icon
    as={PiPlusBold}
    boxSize={6}
    transitionProperty="common"
    transitionDuration="normal"
    color="base.400"
    onClick={handleCreateBoard}
    cursor="pointer"
  />
  );
};

export default memo(AddBoardButton);
