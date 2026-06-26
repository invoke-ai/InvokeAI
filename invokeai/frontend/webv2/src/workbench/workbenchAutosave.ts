import type { WorkbenchState } from './types';

export const getPersistedStateKey = (state: WorkbenchState): string =>
  JSON.stringify({
    account: state.account,
    activeProjectId: state.activeProjectId,
    errorLog: state.errorLog,
    projects: state.projects,
    widgetFailures: state.widgetFailures,
  });

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
  completedStateKey,
  currentPersistedRevision,
  currentStateKey,
}: {
  completedPersistedRevision: number;
  completedStateKey: string;
  currentPersistedRevision: number;
  currentStateKey: string;
}): boolean => completedPersistedRevision === currentPersistedRevision && completedStateKey === currentStateKey;
