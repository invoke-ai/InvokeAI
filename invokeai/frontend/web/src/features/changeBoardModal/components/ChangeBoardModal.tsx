import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvConfirmationAlertDialog } from 'common/components/InvConfirmationAlertDialog/InvConfirmationAlertDialog';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { InvText } from 'common/components/InvText/wrapper';
import {
  changeBoardReset,
  isModalOpenChanged,
} from 'features/changeBoardModal/store/slice';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import {
  useAddImagesToBoardMutation,
  useRemoveImagesFromBoardMutation,
} from 'services/api/endpoints/images';

const selector = createMemoizedSelector(
  [stateSelector],
  ({ changeBoardModal }) => {
    const { isModalOpen, imagesToChange } = changeBoardModal;

    return {
      isModalOpen,
      imagesToChange,
    };
  }
);

const ChangeBoardModal = () => {
  const dispatch = useAppDispatch();
  const [selectedBoard, setSelectedBoard] = useState<string | null>();
  const { data: boards, isFetching } = useListAllBoardsQuery();
  const { imagesToChange, isModalOpen } = useAppSelector(selector);
  const [addImagesToBoard] = useAddImagesToBoardMutation();
  const [removeImagesFromBoard] = useRemoveImagesFromBoardMutation();
  const { t } = useTranslation();

  const options = useMemo<InvSelectOption[]>(() => {
    return [{ label: t('boards.uncategorized'), value: 'none' }].concat(
      (boards ?? []).map((board) => ({
        label: board.board_name,
        value: board.board_id,
      }))
    );
  }, [boards, t]);

  const value = useMemo(
    () => options.find((o) => o.value === selectedBoard),
    [options, selectedBoard]
  );

  const handleClose = useCallback(() => {
    dispatch(changeBoardReset());
    dispatch(isModalOpenChanged(false));
  }, [dispatch]);

  const handleChangeBoard = useCallback(() => {
    if (!imagesToChange.length || !selectedBoard) {
      return;
    }

    if (selectedBoard === 'none') {
      removeImagesFromBoard({ imageDTOs: imagesToChange });
    } else {
      addImagesToBoard({
        imageDTOs: imagesToChange,
        board_id: selectedBoard,
      });
    }
    setSelectedBoard(null);
    dispatch(changeBoardReset());
  }, [
    addImagesToBoard,
    dispatch,
    imagesToChange,
    removeImagesFromBoard,
    selectedBoard,
  ]);

  const onChange = useCallback<InvSelectOnChange>((v) => {
    if (!v) {
      return;
    }
    setSelectedBoard(v.value);
  }, []);

  return (
    <InvConfirmationAlertDialog
      isOpen={isModalOpen}
      onClose={handleClose}
      title={t('boards.changeBoard')}
      acceptCallback={handleChangeBoard}
      acceptButtonText={t('boards.move')}
      cancelButtonText={t('boards.cancel')}
    >
      <Flex flexDir="column" gap={4}>
        <InvText>
          {t('boards.movingImagesToBoard', {
            count: imagesToChange.length,
          })}
          :
        </InvText>
        <InvControl isDisabled={isFetching}>
          <InvSelect
            placeholder={
              isFetching ? t('boards.loading') : t('boards.selectBoard')
            }
            onChange={onChange}
            value={value}
            options={options}
          />
        </InvControl>
      </Flex>
    </InvConfirmationAlertDialog>
  );
};

export default memo(ChangeBoardModal);
