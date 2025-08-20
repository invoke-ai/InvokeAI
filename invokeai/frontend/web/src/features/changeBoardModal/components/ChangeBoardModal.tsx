import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, ConfirmationAlertDialog, Flex, FormControl, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
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
import { useAddVideosToBoardMutation, useRemoveVideosFromBoardMutation } from 'services/api/endpoints/videos';

const selectImagesToChange = createSelector(
  selectChangeBoardModalSlice,
  (changeBoardModal) => changeBoardModal.image_names
);

const selectVideosToChange = createSelector(
  selectChangeBoardModalSlice,
  (changeBoardModal) => changeBoardModal.video_ids
);

const selectIsModalOpen = createSelector(
  selectChangeBoardModalSlice,
  (changeBoardModal) => changeBoardModal.isModalOpen
);

const ChangeBoardModal = () => {
  useAssertSingleton('ChangeBoardModal');
  const dispatch = useAppDispatch();
  const currentBoardId = useAppSelector(selectSelectedBoardId);
  const [selectedBoardId, setSelectedBoardId] = useState<string | null>();
  const { data: boards, isFetching } = useListAllBoardsQuery({ include_archived: true });
  const isModalOpen = useAppSelector(selectIsModalOpen);
  const imagesToChange = useAppSelector(selectImagesToChange);
  const videosToChange = useAppSelector(selectVideosToChange);
  const [addImagesToBoard] = useAddImagesToBoardMutation();
  const [removeImagesFromBoard] = useRemoveImagesFromBoardMutation();
  const [addVideosToBoard] = useAddVideosToBoardMutation();
  const [removeVideosFromBoard] = useRemoveVideosFromBoardMutation();
  const { t } = useTranslation();

  const options = useMemo<ComboboxOption[]>(() => {
    return [{ label: t('boards.uncategorized'), value: 'none' }]
      .concat(
        (boards ?? [])
          .map((board) => ({
            label: board.board_name,
            value: board.board_id,
          }))
          .sort((a, b) => a.label.localeCompare(b.label))
      )
      .filter((board) => board.value !== currentBoardId);
  }, [boards, currentBoardId, t]);

  const value = useMemo(() => options.find((o) => o.value === selectedBoardId), [options, selectedBoardId]);

  const handleClose = useCallback(() => {
    dispatch(changeBoardReset());
    dispatch(isModalOpenChanged(false));
  }, [dispatch]);

  const handleChangeBoard = useCallback(() => {
    if (!selectedBoard || (imagesToChange.length === 0 && videosToChange.length === 0)) {
      return;
    }

    if (imagesToChange.length) {
      if (selectedBoard === 'none') {
        removeImagesFromBoard({ image_names: imagesToChange });
      } else {
        addImagesToBoard({
          image_names: imagesToChange,
          board_id: selectedBoard,
        });
      }
    }
    if (videosToChange.length) {
      if (selectedBoard === 'none') {
        removeVideosFromBoard({ video_ids: videosToChange });
      } else {
        addVideosToBoard({
          video_ids: videosToChange,
          board_id: selectedBoard,
        });
      }
    }
    dispatch(changeBoardReset());
  }, [addImagesToBoard, dispatch, imagesToChange, videosToChange, removeImagesFromBoard, selectedBoard]);

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
          {imagesToChange.length > 0 && t('boards.movingImagesToBoard', {
            count: imagesToChange.length,
          })}
          {videosToChange.length > 0 && t('boards.movingVideosToBoard', {
            count: videosToChange.length,
          })}
          :
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
