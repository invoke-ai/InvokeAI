interface AutosaveScheduleDecisionInput {
  failedPersistedRevision: number | null;
  lastSavedPersistedRevision: number;
  persistedRevision: number;
  scheduledPersistedRevision: number | null;
}

interface AutosaveScheduleDecision {
  failedPersistedRevision: number | null;
  shouldSchedule: boolean;
}

export const getAutosaveScheduleDecision = ({
  failedPersistedRevision,
  lastSavedPersistedRevision,
  persistedRevision,
  scheduledPersistedRevision,
}: AutosaveScheduleDecisionInput): AutosaveScheduleDecision => {
  if (persistedRevision === lastSavedPersistedRevision) {
    return { failedPersistedRevision: null, shouldSchedule: false };
  }

  if (persistedRevision === scheduledPersistedRevision) {
    return { failedPersistedRevision, shouldSchedule: false };
  }

  if (persistedRevision === failedPersistedRevision) {
    return { failedPersistedRevision, shouldSchedule: false };
  }

  return { failedPersistedRevision: null, shouldSchedule: true };
};

export const isAutosaveCompletionCurrent = ({
  completedPersistedRevision,
  currentPersistedRevision,
}: {
  completedPersistedRevision: number;
  currentPersistedRevision: number;
}): boolean => completedPersistedRevision === currentPersistedRevision;
