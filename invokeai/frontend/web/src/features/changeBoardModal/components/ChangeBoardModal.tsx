import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Flex,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import {
  useAddImagesToBoardMutation,
  useRemoveImagesFromBoardMutation,
} from 'services/api/endpoints/images';
import { changeBoardReset, isModalOpenChanged } from '../store/slice';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector],
  ({ changeBoardModal }) => {
    const { isModalOpen, imagesToChange } = changeBoardModal;

    return {
      isModalOpen,
      imagesToChange,
    };
  },
  defaultSelectorOptions
);

const ChangeBoardModal = () => {
  const dispatch = useAppDispatch();
  const [selectedBoard, setSelectedBoard] = useState<string | null>();
  const { data: boards, isFetching } = useListAllBoardsQuery();
  const { imagesToChange, isModalOpen } = useAppSelector(selector);
  const [addImagesToBoard] = useAddImagesToBoardMutation();
  const [removeImagesFromBoard] = useRemoveImagesFromBoardMutation();
  const { t } = useTranslation();

  const data = useMemo(() => {
    const data: { label: string; value: string }[] = [
      { label: t('boards.uncategorized'), value: 'none' },
    ];
    (boards ?? []).forEach((board) =>
      data.push({
        label: board.board_name,
        value: board.board_id,
      })
    );

    return data;
  }, [boards, t]);

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

  const handleSetSelectedBoard = useCallback(
    (v: string | null) => setSelectedBoard(v),
    []
  );

  const cancelRef = useRef<HTMLButtonElement>(null);

  return (
    <AlertDialog
      isOpen={isModalOpen}
      onClose={handleClose}
      leastDestructiveRef={cancelRef}
      isCentered
    >
      <AlertDialogOverlay>
        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('boards.changeBoard')}
          </AlertDialogHeader>

          <AlertDialogBody>
            <Flex sx={{ flexDir: 'column', gap: 4 }}>
              <Text>
                {t('boards.moving')} {`${imagesToChange.length}`}{' '}
                {t('gallery.image')}
                {`${imagesToChange.length > 1 ? 's' : ''}`}{' '}
                {t('boards.toBoard')}:
              </Text>
              <IAIMantineSearchableSelect
                placeholder={
                  isFetching ? t('boards.loading') : t('boards.selectBoard')
                }
                disabled={isFetching}
                onChange={handleSetSelectedBoard}
                value={selectedBoard}
                data={data}
              />
            </Flex>
          </AlertDialogBody>
          <AlertDialogFooter>
            <IAIButton ref={cancelRef} onClick={handleClose}>
              {t('boards.cancel')}
            </IAIButton>
            <IAIButton colorScheme="accent" onClick={handleChangeBoard} ml={3}>
              {t('boards.move')}
            </IAIButton>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(ChangeBoardModal);
