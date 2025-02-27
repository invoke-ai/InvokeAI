import { buildUseDisclosure } from 'common/hooks/useBoolean';

/**
 * Tracks the state for the workflow library modal.
 */
export const [useWorkflowLibraryModal, $isWorkflowLibraryModalOpen] = buildUseDisclosure(false);
