interface AutosaveScheduleDecisionInput {
  failedStateKey: string | null;
  lastSavedStateKey: string;
  persistedStateKey: string;
  scheduledStateKey: string | null;
}

interface AutosaveScheduleDecision {
  failedStateKey: string | null;
  shouldSchedule: boolean;
}

export const getAutosaveScheduleDecision = ({
  failedStateKey,
  lastSavedStateKey,
  persistedStateKey,
  scheduledStateKey,
}: AutosaveScheduleDecisionInput): AutosaveScheduleDecision => {
  if (persistedStateKey === lastSavedStateKey) {
    return { failedStateKey: null, shouldSchedule: false };
  }

  if (persistedStateKey === scheduledStateKey) {
    return { failedStateKey, shouldSchedule: false };
  }

  if (persistedStateKey === failedStateKey) {
    return { failedStateKey, shouldSchedule: false };
  }

  return { failedStateKey: null, shouldSchedule: true };
};
