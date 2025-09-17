import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, ConfirmationAlertDialog, Flex, FormControl, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useChangeBoardModalApi, useChangeBoardModalState } from 'features/changeBoardModal/store/state';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useAddImagesToBoardMutation, useRemoveImagesFromBoardMutation } from 'services/api/endpoints/images';
import { useAddVideosToBoardMutation, useRemoveVideosFromBoardMutation } from 'services/api/endpoints/videos';

const ChangeBoardModal = () => {
  useAssertSingleton('ChangeBoardModal');
  const currentBoardId = useAppSelector(selectSelectedBoardId);
  const [selectedBoardId, setSelectedBoardId] = useState<string | null>();
  const { data: boards, isFetching } = useListAllBoardsQuery({ include_archived: true });
  const changeBoardModalState = useChangeBoardModalState();
  const changeBoardModal = useChangeBoardModalApi();
  const imagesToChange = changeBoardModalState.imageNames;
  const videosToChange = changeBoardModalState.videoIds;
  const isModalOpen = changeBoardModalState.isOpen;
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
    changeBoardModal.close();
  }, [changeBoardModal]);

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
      if (selectedBoardId === 'none') {
        removeVideosFromBoard({ video_ids: videosToChange });
      } else {
        addVideosToBoard({
          video_ids: videosToChange,
          board_id: selectedBoardId,
        });
      }
    }
    changeBoardModal.close();
  }, [
    addImagesToBoard,
    changeBoardModal,
    imagesToChange,
    videosToChange,
    removeImagesFromBoard,
    selectedBoardId,
    addVideosToBoard,
    removeVideosFromBoard,
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
          {imagesToChange.length > 0 &&
            t('boards.movingImagesToBoard', {
              count: imagesToChange.length,
            })}
          {videosToChange.length > 0 &&
            t('boards.movingVideosToBoard', {
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
