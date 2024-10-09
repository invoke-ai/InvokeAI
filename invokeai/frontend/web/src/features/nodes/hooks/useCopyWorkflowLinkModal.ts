import { useStore } from '@nanostores/react';
import { atom } from 'nanostores';
import { useCallback } from 'react';

const $isOpen = atom<boolean>(false);

export const useCopyWorkflowLinkModal = () => {
  const isOpen = useStore($isOpen);
  const onOpen = useCallback(() => {
    $isOpen.set(true);
  }, []);
  const onClose = useCallback(() => {
    $isOpen.set(false);
  }, []);

  return { isOpen, onOpen, onClose };
};
