import { Editor } from 'features/editImageModal/lib/editor';
import { atom } from 'nanostores';

type EditImageModalState =
  | {
      isOpen: false;
      imageName: null;
      editor: null;
    }
  | {
      isOpen: true;
      imageName: string;
      editor: Editor;
    };

export const $editImageModalState = atom<EditImageModalState>({
  isOpen: false,
  imageName: null,
  editor: null,
});

export const openEditImageModal = (imageName: string) => {
  $editImageModalState.set({
    isOpen: true,
    imageName,
    editor: new Editor(),
  });
};

export const closeEditImageModal = () => {
  const { editor } = $editImageModalState.get();
  editor?.destroy();
  $editImageModalState.set({
    isOpen: false,
    imageName: null,
    editor: null,
  });
};
