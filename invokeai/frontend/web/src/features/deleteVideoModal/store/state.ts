import { useStore } from '@nanostores/react';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { intersection } from 'es-toolkit/compat';
import { selectGetVideoIdsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { itemSelected } from 'features/gallery/store/gallerySlice';
import { selectSystemShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { atom } from 'nanostores';
import { useMemo } from 'react';
import { videosApi } from 'services/api/endpoints/videos';

// Implements an awaitable modal dialog for deleting images

type DeleteVideosModalState = {
  video_ids: string[];
  isOpen: boolean;
  resolve?: () => void;
  reject?: (reason?: string) => void;
};

const getInitialState = (): DeleteVideosModalState => ({
  video_ids: [],
  isOpen: false,
});

const $deleteVideosModalState = atom<DeleteVideosModalState>(getInitialState());

const deleteVideosWithDialog = async (video_ids: string[], store: AppStore): Promise<void> => {
  const { getState } = store;
  const shouldConfirmOnDelete = selectSystemShouldConfirmOnDelete(getState());

  if (!shouldConfirmOnDelete) {
    // If we don't need to confirm and the resources are not in use, delete them directly
    await handleDeletions(video_ids, store);
    return;
  }

  return new Promise<void>((resolve, reject) => {
    $deleteVideosModalState.set({
      video_ids,
      isOpen: true,
      resolve,
      reject,
    });
  });
};

const handleDeletions = async (video_ids: string[], store: AppStore) => {
  try {
    const { dispatch, getState } = store;
    const state = getState();
    const { data } = videosApi.endpoints.getVideoIds.select(selectGetVideoIdsQueryArgs(state))(state);
    const index = data?.video_ids.findIndex((id) => id === video_ids[0]);
    const { deleted_videos } = await dispatch(
      videosApi.endpoints.deleteVideos.initiate({ video_ids }, { track: false })
    ).unwrap();

    const newVideoIds = data?.video_ids.filter((id) => !deleted_videos.includes(id)) || [];
    const newSelectedVideoId = newVideoIds[index ?? 0] || null;

    if (
      intersection(
        state.gallery.selection.map((s) => s.id),
        video_ids
      ).length > 0 &&
      newSelectedVideoId
    ) {
      // Some selected images were deleted, clear selection
      dispatch(itemSelected({ type: 'video', id: newSelectedVideoId }));
    }
  } catch {
    // no-op
  }
};

const confirmDeletion = async (store: AppStore) => {
  const state = $deleteVideosModalState.get();
  await handleDeletions(state.video_ids, store);
  state.resolve?.();
  closeSilently();
};

const cancelDeletion = () => {
  const state = $deleteVideosModalState.get();
  state.reject?.('User canceled');
  closeSilently();
};

const closeSilently = () => {
  $deleteVideosModalState.set(getInitialState());
};

export const useDeleteVideoModalState = () => {
  const state = useStore($deleteVideosModalState);
  return state;
};

export const useDeleteVideoModalApi = () => {
  const store = useAppStore();
  const api = useMemo(
    () => ({
      delete: (video_ids: string[]) => deleteVideosWithDialog(video_ids, store),
      confirm: () => confirmDeletion(store),
      cancel: cancelDeletion,
      close: closeSilently,
    }),
    [store]
  );

  return api;
};
