import { atom } from 'nanostores';

type SystemPromptsModalState = {
  isOpen: boolean;
  editingId: string | null; // null + isOpen = list view; non-null = edit view; 'new' = create view
};

export const $systemPromptsModalState = atom<SystemPromptsModalState>({
  isOpen: false,
  editingId: null,
});

export const openSystemPromptsModal = () => {
  $systemPromptsModalState.set({ isOpen: true, editingId: null });
};

export const closeSystemPromptsModal = () => {
  $systemPromptsModalState.set({ isOpen: false, editingId: null });
};

export const showSystemPromptsList = () => {
  $systemPromptsModalState.set({ isOpen: true, editingId: null });
};

export const showSystemPromptEditor = (id: string | 'new') => {
  $systemPromptsModalState.set({ isOpen: true, editingId: id });
};
