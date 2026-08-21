import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, ConfirmationAlertDialog, Flex, FormControl, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { captureAuthContext, isSameAuthContext } from 'features/auth/store/authTokenRefresh';
import {
  canRetainFailedSelection,
  changeBoardReset,
  imagesToChangeSelected,
  isModalOpenChanged,
  selectChangeBoardModalSlice,
  videosToChangeSelected,
} from 'features/changeBoardModal/store/slice';
import { selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { toast } from 'features/toast/toast';
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
  const store = useAppStore();
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

  const handleChangeBoard = useCallback(async () => {
    if (!selectedBoardId || (imagesToChange.length === 0 && videosToChange.length === 0)) {
      return;
    }

    const authContext = captureAuthContext();

    // Awaited, not fired and forgotten. The batch routes report per-name failures in
    // `failed_images`, and accepting this dialog resets the selection on the way out
    // (ConfirmationAlertDialog calls acceptCallback then onClose) — so the names that did not
    // move used to be dropped along with the ones that did, leaving nothing to retry from.
    //
    // Per-name failures and whole-request failures are toasted by the endpoint itself. What is
    // left to do here is keep names that need a retry selected.
    const operationId = selectChangeBoardModalSlice(store.getState()).operation_id;
    const failedImageNamesPromise: Promise<string[]> = !imagesToChange.length
      ? Promise.resolve([])
      : (selectedBoardId === 'none'
          ? removeImagesFromBoard({ image_names: imagesToChange })
          : addImagesToBoard({ image_names: imagesToChange, board_id: selectedBoardId })
        )
          .unwrap()
          .then((result) => result.failed_images)
          .catch(() => imagesToChange);

    const videoMutations: { videoName: string; promise: Promise<unknown> }[] = [];
    if (videosToChange.length) {
      // The video board endpoints take one video at a time; the context menu acts on a single
      // selection, so this is normally a one-iteration loop.
      for (const video_name of videosToChange) {
        if (selectedBoardId === 'none') {
          videoMutations.push({ videoName: video_name, promise: removeVideoFromBoard({ video_name }).unwrap() });
        } else {
          videoMutations.push({
            videoName: video_name,
            promise: addVideoToBoard({ board_id: selectedBoardId, video_name }).unwrap(),
          });
        }
      }
    }

    // Both kinds go out together: the video routes take one name at a time, and serializing
    // them behind the image batch would leave a large move waiting on the other's round trips.
    const [failedImageNames, results] = await Promise.all([
      failedImageNamesPromise,
      Promise.allSettled(videoMutations.map(({ promise }) => promise)),
    ]);
    const failed = results.filter((result) => result.status === 'rejected');
    const isSameSession = isSameAuthContext(authContext);

    // Reported ahead of the ownership guard below, not behind it. Nothing else reports a move
    // made from this dialog: the video board routes carry no `onQueryStarted` and no
    // `matchRejected` listener, unlike the image batch routes, and the one other emitter of this
    // toast id — `settleVideoBoardMutations`, on the drag-and-drop path — only ever settles the
    // mutations it fired itself. So behind the guard, opening and cancelling any second dialog
    // while this move was in flight would leave the user with no notice at all that it failed.
    // The guard exists to keep a stale write out of a shared slice, not to decide who gets told
    // about a request they themselves started. Only the session check applies here: the failure
    // belongs to whoever started the move, so it is not raised at whoever holds the tab after a
    // logout.
    if (failed.length > 0 && isSameSession) {
      toast({
        id: 'VIDEOS_FAILED_TO_MOVE',
        title: t('toast.videosFailedToMove', { count: failed.length }),
        status: 'warning',
      });
    }

    // Checked before *any* of the writes below, the reset included: all of them land after an
    // unbounded await, and the reset is as capable of clearing a selection that now belongs to
    // someone else as the retain is of overwriting it.
    if (!canRetainFailedSelection(selectChangeBoardModalSlice(store.getState()), operationId, isSameSession)) {
      return;
    }
    if (failed.length === 0 && failedImageNames.length === 0) {
      dispatch(changeBoardReset());
      return;
    }
    // At most one of these fires: the two selections are mutually exclusive by construction —
    // imagesToChangeSelected clears video_names and videosToChangeSelected clears image_names,
    // and every caller opens this dialog through one of them. Reopen the dialog so retained
    // failures are actionable instead of inert state after the accept close.
    if (failedImageNames.length > 0) {
      dispatch(imagesToChangeSelected(failedImageNames));
      dispatch(isModalOpenChanged(true));
    }
    if (failed.length === 0) {
      return;
    }
    const failedVideoNames = results.flatMap((result, index) =>
      result.status === 'rejected' && videoMutations[index] ? [videoMutations[index].videoName] : []
    );
    dispatch(videosToChangeSelected(failedVideoNames));
    dispatch(isModalOpenChanged(true));
  }, [
    addImagesToBoard,
    addVideoToBoard,
    dispatch,
    imagesToChange,
    removeImagesFromBoard,
    removeVideoFromBoard,
    selectedBoardId,
    store,
    t,
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
