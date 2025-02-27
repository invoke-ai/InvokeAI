import { buildUseDisclosure } from 'common/hooks/useBoolean';

/**
 * Tracks the state for the workflow list menu.
 */
export const [useWorkflowListMenu, $isWorkflowListMenuIsOpen] = buildUseDisclosure(false);
