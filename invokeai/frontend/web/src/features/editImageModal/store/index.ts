import type { Editor } from 'features/editImageModal/lib/editor';
import { atom } from 'nanostores';

export type EditImageModalState = {
  editor: Editor;
  onApplyCrop: () => Promise<void> | void;
  onReady: () => Promise<void> | void;
};

export const $editImageModalState = atom<EditImageModalState | null>(null);

export const openEditImageModal = (state: EditImageModalState) => {
  $editImageModalState.set(state);
};

export const closeEditImageModal = () => {
  const state = $editImageModalState.get();
  state?.editor.destroy();
  $editImageModalState.set(null);
};
