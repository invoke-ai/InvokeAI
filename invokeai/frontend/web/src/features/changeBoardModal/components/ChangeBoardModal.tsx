import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, ConfirmationAlertDialog, Flex, FormControl, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import {
  changeBoardReset,
  isModalOpenChanged,
  selectChangeBoardModalSlice,
} from 'features/changeBoardModal/store/slice';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useAddImagesToBoardMutation, useRemoveImagesFromBoardMutation } from 'services/api/endpoints/images';
import { useAddVideoToBoardMutation, useRemoveVideoFromBoardMutation } from 'services/api/endpoints/videos';
import type { BoardDTO } from 'services/api/types';

const selectImagesToChange = createSelector(
  selectChangeBoardModalSlice,
  (changeBoardModal) => changeBoardModal.image_names
);

const selectVideosToChange = createSelector(
  selectChangeBoardModalSlice,
  (changeBoardModal) => changeBoardModal.video_names
);

const selectIsModalOpen = createSelector(
  selectChangeBoardModalSlice,
  (changeBoardModal) => changeBoardModal.isModalOpen
);

const ChangeBoardModal = () => {
  useAssertSingleton('ChangeBoardModal');
  const dispatch = useAppDispatch();
  const currentBoardId = useAppSelector(selectSelectedBoardId);
  const currentUser = useAppSelector(selectCurrentUser);
  const [selectedBoardId, setSelectedBoardId] = useState<string | null>();
  const { data: boards, isFetching } = useListAllBoardsQuery({ include_archived: true });
  const isModalOpen = useAppSelector(selectIsModalOpen);
  const imagesToChange = useAppSelector(selectImagesToChange);
  const videosToChange = useAppSelector(selectVideosToChange);
  const [addImagesToBoard] = useAddImagesToBoardMutation();
  const [removeImagesFromBoard] = useRemoveImagesFromBoardMutation();
  const [addVideoToBoard] = useAddVideoToBoardMutation();
  const [removeVideoFromBoard] = useRemoveVideoFromBoardMutation();
  const { t } = useTranslation();

  // Returns true if the current user can write images to the given board.
  const canWriteToBoard = useCallback(
    (board: BoardDTO): boolean => {
      const isOwnerOrAdmin = !currentUser || currentUser.is_admin || board.user_id === currentUser.user_id;
      return isOwnerOrAdmin || board.board_visibility === 'public';
    },
    [currentUser]
  );

  const options = useMemo<ComboboxOption[]>(() => {
    return [{ label: t('boards.uncategorized'), value: 'none' }]
      .concat(
        (boards ?? [])
          .filter(canWriteToBoard)
          .map((board) => ({
            label: board.board_name,
            value: board.board_id,
          }))
          .sort((a, b) => a.label.localeCompare(b.label))
      )
      .filter((board) => board.value !== currentBoardId);
  }, [boards, canWriteToBoard, currentBoardId, t]);

  const value = useMemo(() => options.find((o) => o.value === selectedBoardId), [options, selectedBoardId]);

  const handleClose = useCallback(() => {
    dispatch(changeBoardReset());
    dispatch(isModalOpenChanged(false));
  }, [dispatch]);

  const handleChangeBoard = useCallback(() => {
    if (!selectedBoardId || (imagesToChange.length === 0 && videosToChange.length === 0)) {
      return;
    }

    if (imagesToChange.length) {
      if (selectedBoardId === 'none') {
        removeImagesFromBoard({ image_names: imagesToChange });
      } else {
        addImagesToBoard({
          image_names: imagesToChange,
          board_id: selectedBoardId,
        });
      }
    }

    if (videosToChange.length) {
      // The video board endpoints take one video at a time; the context menu acts on a single
      // selection, so this is normally a one-iteration loop.
      for (const video_name of videosToChange) {
        if (selectedBoardId === 'none') {
          removeVideoFromBoard({ video_name });
        } else {
          addVideoToBoard({ board_id: selectedBoardId, video_name });
        }
      }
    }

    dispatch(changeBoardReset());
  }, [
    addImagesToBoard,
    addVideoToBoard,
    dispatch,
    imagesToChange,
    removeImagesFromBoard,
    removeVideoFromBoard,
    selectedBoardId,
    videosToChange,
  ]);

  const onChange = useCallback<ComboboxOnChange>((v) => {
    if (!v) {
      return;
    }
    setSelectedBoardId(v.value);
  }, []);

  return (
    <ConfirmationAlertDialog
      isOpen={isModalOpen}
      onClose={handleClose}
      title={t('boards.changeBoard')}
      acceptCallback={handleChangeBoard}
      acceptButtonText={t('boards.move')}
      cancelButtonText={t('boards.cancel')}
      useInert={false}
    >
      <Flex flexDir="column" gap={4}>
        <Text>
          {t('boards.movingImagesToBoard', {
            count: imagesToChange.length + videosToChange.length,
          })}
        </Text>
        <FormControl isDisabled={isFetching}>
          <Combobox
            placeholder={isFetching ? t('boards.loading') : t('boards.selectBoard')}
            onChange={onChange}
            value={value}
            options={options}
          />
        </FormControl>
      </Flex>
    </ConfirmationAlertDialog>
  );
};

export default memo(ChangeBoardModal);
