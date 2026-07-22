import { useStore } from '@nanostores/react';
import type { AppDispatch, AppStore, RootState } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { intersection } from 'es-toolkit/compat';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import {
  pickSelectionAfterDelete,
  selectCachedGalleryItemNames,
} from 'features/gallery/store/selectCachedGalleryItemNames';
import { fieldVideoValueChanged } from 'features/nodes/store/nodesSlice';
import { isVideoFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { selectSystemShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { atom } from 'nanostores';
import { useMemo } from 'react';
import { videosApi } from 'services/api/endpoints/videos';

// Parallel of features/deleteImageModal/store/state.ts but trimmed: videos don't show up
// on canvas layers, ref-image entities, or upscale inputs the way images do, so there is
// no "usage" analysis to compute and the dialog is a straight confirm. Workflow nodes DO
// take VideoField inputs, though — confirmed deletions must clear those references.

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

export const clearNodesVideoFields = (state: RootState, dispatch: AppDispatch, video_name: string) => {
  state.nodes.present.nodes.forEach((node) => {
    if (!isInvocationNode(node)) {
      return;
    }
    for (const input of Object.values(node.data.inputs)) {
      if (isVideoFieldInputInstance(input) && input.value?.video_name === video_name) {
        dispatch(
          fieldVideoValueChanged({
            nodeId: node.data.id,
            fieldName: input.name,
            value: undefined,
          })
        );
      }
    }
  });
};

export const handleDeletions = async (video_names: string[], store: AppStore) => {
  const { dispatch, getState } = store;

  // Snapshot the polymorphic gallery list and the currently-displayed item *before* the
  // delete fires; once the network call resolves the cache will already have shifted.
  const stateBefore = getState();
  const galleryItemNames = selectCachedGalleryItemNames(stateBefore);
  const lastSelected = selectLastSelectedItem(stateBefore);
  const lastSelectedIndex =
    lastSelected && video_names.includes(lastSelected) ? galleryItemNames.indexOf(lastSelected) : -1;

  // One batch request; the backend skips per-video failures and reports what actually
  // got deleted, so partial failures don't discard the successes.
  let deletedNames = new Set<string>();
  try {
    const result = await dispatch(
      videosApi.endpoints.deleteVideos.initiate({ video_names }, { track: false })
    ).unwrap();
    deletedNames = new Set(result.deleted_videos);
    if (result.failed_videos.length > 0) {
      toast({
        status: 'warning',
        title: t('toast.videoDeleteFailed'),
        description: t('toast.videoDeletePartial', { count: result.failed_videos.length }),
      });
    }
  } catch {
    // The whole request failed — nothing was confirmed deleted, so leave selection and
    // node references untouched. The mutation is untracked, so this toast is the only
    // user-visible signal that the delete didn't happen.
    toast({
      status: 'error',
      title: t('toast.videoDeleteFailed'),
      description: t('toast.videoDeleteFailedDesc'),
    });
  }

  // Clear workflow-node VideoField inputs that referenced a now-deleted video. Failed
  // deletions keep their references — those videos still exist.
  for (const video_name of deletedNames) {
    clearNodesVideoFields(getState(), dispatch, video_name);
  }

  // If anything in the active selection was actually deleted, advance to a still-living
  // neighbour (prev > next) so the Viewer doesn't drop to its empty-state placeholder.
  // Failed deletions are excluded: those videos still exist, so the selection must not
  // jump away from them, and they remain valid replacement candidates.
  const stateAfter = getState();
  if (intersection(stateAfter.gallery.selection, [...deletedNames]).length > 0) {
    if (lastSelected && !deletedNames.has(lastSelected)) {
      // The displayed item survived (its delete failed) — keep viewing it and just prune
      // the deleted items from the multi-selection.
      dispatch(imageSelected(lastSelected));
    } else {
      const replacement =
        lastSelectedIndex >= 0 ? pickSelectionAfterDelete(galleryItemNames, lastSelectedIndex, deletedNames) : null;
      dispatch(imageSelected(replacement));
    }
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
