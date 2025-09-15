import type { Editor } from 'features/editImageModal/lib/editor';
import { atom } from 'nanostores';

type EditImageModalState =
  | {
      isOpen: false;
      editor: null;
    }
  | {
      isOpen: true;
      editor: Editor;
    };

export const $editImageModalState = atom<EditImageModalState>({
  isOpen: false,
  editor: null,
});

export const openEditImageModal = (editor: Editor) => {
  $editImageModalState.set({
    isOpen: true,
    editor,
  });
};

export const closeEditImageModal = () => {
  const { editor } = $editImageModalState.get();
  editor?.destroy();
  $editImageModalState.set({
    isOpen: false,
    editor: null,
  });
};
