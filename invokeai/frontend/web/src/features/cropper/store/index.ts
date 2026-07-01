import type { Editor } from 'features/cropper/lib/editor';
import { atom } from 'nanostores';

export type CropImageModalState = {
  editor: Editor;
  onApplyCrop: () => Promise<void> | void;
  onReady: () => Promise<void> | void;
};

const $state = atom<CropImageModalState | null>(null);

const open = (state: CropImageModalState) => {
  $state.set(state);
};

const close = () => {
  const state = $state.get();
  state?.editor.destroy();
  $state.set(null);
};

export const cropImageModalApi = {
  $state,
  open,
  close,
};
