import { useStore } from '@nanostores/react';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { intersection } from 'es-toolkit/compat';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { selectSystemShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { atom } from 'nanostores';
import { useMemo } from 'react';
import { videosApi } from 'services/api/endpoints/videos';

// Parallel of features/deleteImageModal/store/state.ts but trimmed: videos don't show up
// on canvas layers, node fields, ref-image entities, or upscale inputs the way images do,
// so there is no "usage" analysis to compute. The dialog is a straight confirm.

type DeleteVideosModalState = {
  video_names: string[];
  isOpen: boolean;
  resolve?: () => void;
  reject?: (reason?: string) => void;
};

const getInitialState = (): DeleteVideosModalState => ({
  video_names: [],
  isOpen: false,
});

const $deleteModalState = atom<DeleteVideosModalState>(getInitialState());

const deleteVideosWithDialog = async (video_names: string[], store: AppStore): Promise<void> => {
  const { getState } = store;
  const shouldConfirmOnDelete = selectSystemShouldConfirmOnDelete(getState());

  if (!shouldConfirmOnDelete) {
    await handleDeletions(video_names, store);
    return;
  }

  return new Promise<void>((resolve, reject) => {
    $deleteModalState.set({
      video_names,
      isOpen: true,
      resolve,
      reject,
    });
  });
};

const handleDeletions = async (video_names: string[], store: AppStore) => {
  const { dispatch, getState } = store;
  // The backend exposes single-video DELETE today; loop here so the API surface for callers
  // stays "give me a list, I'll handle it" and a future bulk endpoint can be slotted in
  // without touching call sites.
  for (const video_name of video_names) {
    try {
      await dispatch(videosApi.endpoints.deleteVideo.initiate({ video_name }, { track: false })).unwrap();
    } catch {
      // Continue with the rest of the batch — partial failures shouldn't leave the user
      // with a broken modal state.
    }
  }

  // Clear the deleted videos from the current selection so the viewer doesn't try to
  // resolve a name that no longer has a DTO.
  const state = getState();
  if (intersection(state.gallery.selection, video_names).length > 0) {
    dispatch(imageSelected(null));
  }
};

const confirmDeletion = async (store: AppStore) => {
  const state = $deleteModalState.get();
  await handleDeletions(state.video_names, store);
  state.resolve?.();
  closeSilently();
};

const cancelDeletion = () => {
  const state = $deleteModalState.get();
  state.reject?.('User canceled');
  closeSilently();
};

const closeSilently = () => {
  $deleteModalState.set(getInitialState());
};

export const useDeleteVideoModalState = () => useStore($deleteModalState);

export const useDeleteVideoModalApi = () => {
  const store = useAppStore();
  return useMemo(
    () => ({
      delete: (video_names: string[]) => deleteVideosWithDialog(video_names, store),
      confirm: () => confirmDeletion(store),
      cancel: cancelDeletion,
      close: closeSilently,
    }),
    [store]
  );
};
