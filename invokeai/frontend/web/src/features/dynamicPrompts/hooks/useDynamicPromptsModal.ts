import { useStore } from '@nanostores/react';
import { atom } from 'nanostores';
import { useCallback } from 'react';

const $isDynamicPromptsModalOpen = atom<boolean>(false);

export const useDynamicPromptsModal = () => {
  const isOpen = useStore($isDynamicPromptsModalOpen);
  const onOpen = useCallback(() => {
    console.log('onOpen');
    $isDynamicPromptsModalOpen.set(true);
  }, []);
  const onClose = useCallback(() => {
    console.log('onClose');
    $isDynamicPromptsModalOpen.set(false);
  }, []);

  return { isOpen, onOpen, onClose };
};
