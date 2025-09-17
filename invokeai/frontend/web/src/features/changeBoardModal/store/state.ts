import { useStore } from '@nanostores/react';
import { atom } from 'nanostores';
import { useMemo } from 'react';

type ChangeBoardModalOpenArgs = {
  imageNames?: string[];
  videoIds?: string[];
};

type ChangeBoardModalState = {
  isOpen: boolean;
  imageNames: string[];
  videoIds: string[];
};

const initialState: ChangeBoardModalState = {
  isOpen: false,
  imageNames: [],
  videoIds: [],
};

const $changeBoardModalState = atom<ChangeBoardModalState>(initialState);

const openModal = ({ imageNames = [], videoIds = [] }: ChangeBoardModalOpenArgs = {}) => {
  $changeBoardModalState.set({
    isOpen: true,
    imageNames: [...imageNames],
    videoIds: [...videoIds],
  });
};

const closeModal = () => {
  $changeBoardModalState.set(initialState);
};

const setImageNames = (imageNames: string[]) => {
  const current = $changeBoardModalState.get();
  $changeBoardModalState.set({ ...current, imageNames: [...imageNames] });
};

const setVideoIds = (videoIds: string[]) => {
  const current = $changeBoardModalState.get();
  $changeBoardModalState.set({ ...current, videoIds: [...videoIds] });
};

const resetSelections = () => {
  const current = $changeBoardModalState.get();
  $changeBoardModalState.set({ ...current, imageNames: [], videoIds: [] });
};

export const useChangeBoardModalState = () => {
  return useStore($changeBoardModalState);
};

export const useChangeBoardModalApi = () => {
  return useMemo(
    () => ({
      open: openModal,
      openWithImages: (imageNames: string[]) => openModal({ imageNames }),
      openWithVideos: (videoIds: string[]) => openModal({ videoIds }),
      setImageNames,
      setVideoIds,
      resetSelections,
      close: closeModal,
    }),
    []
  );
};

export type { ChangeBoardModalState };
