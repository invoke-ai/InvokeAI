const SECTION_STATE_KEY = 'formSectionsOpen';
// Keep recent workflow preferences without accumulating every retired working-copy ID in a project.
const MAX_WORKFLOW_SECTION_STATES = 64;

/** Widget workspace values are personal and project-scoped, unlike the shared authored workflow form. */
export const getFormSectionOpenState = (
  workflowValues: Record<string, unknown> | undefined,
  workflowId: string
): Record<string, boolean> => {
  const all = workflowValues?.[SECTION_STATE_KEY];
  const current = isRecord(all) ? all[workflowId] : null;
  if (!isRecord(current)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(current).filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean')
  );
};

export const updateFormSectionOpenState = (
  workflowValues: Record<string, unknown> | undefined,
  workflowId: string,
  sectionId: string,
  open: boolean,
  validSectionIds?: ReadonlySet<string>
): Record<string, unknown> => {
  const stored = workflowValues?.[SECTION_STATE_KEY];
  const all = isRecord(stored) ? stored : {};
  const recentWorkflows = Object.fromEntries(
    Object.entries(all)
      .filter(([id]) => id !== workflowId)
      .slice(-(MAX_WORKFLOW_SECTION_STATES - 1))
  );
  const previous = getFormSectionOpenState(workflowValues, workflowId);
  const retained = validSectionIds
    ? Object.fromEntries(Object.entries(previous).filter(([id]) => validSectionIds.has(id)))
    : previous;
  return {
    [SECTION_STATE_KEY]: {
      ...recentWorkflows,
      [workflowId]: { ...retained, [sectionId]: open },
    },
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
