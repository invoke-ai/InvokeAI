import { buildUseDisclosure } from 'common/hooks/useBoolean';
import { atom } from 'nanostores';

export type WorkflowLibraryContext = { mode: 'workflow' } | { mode: 'canvas'; onSelect: (workflowId: string) => void };

const defaultContext: WorkflowLibraryContext = { mode: 'workflow' };

export const $workflowLibraryContext = atom<WorkflowLibraryContext>(defaultContext);

const [useDisclosure, $isWorkflowLibraryModalOpen] = buildUseDisclosure(false);

export const useWorkflowLibraryModal = () => {
  const disclosure = useDisclosure();
  const { open: baseOpen, close: baseClose, isOpen } = disclosure;

  const open = (context?: WorkflowLibraryContext) => {
    $workflowLibraryContext.set(context ?? defaultContext);
    baseOpen();
  };

  const close = () => {
    baseClose();
    $workflowLibraryContext.set(defaultContext);
  };

  const toggle = () => {
    if (isOpen) {
      close();
    } else {
      open();
    }
  };

  return {
    ...disclosure,
    open,
    close,
    toggle,
  };
};

export { $isWorkflowLibraryModalOpen };
