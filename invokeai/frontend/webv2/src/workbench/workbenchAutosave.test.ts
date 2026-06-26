import { describe, expect, it } from 'vitest';

import { getAutosaveScheduleDecision, isAutosaveCompletionCurrent } from './workbenchAutosave';

describe('getAutosaveScheduleDecision', () => {
  it('does not schedule already-saved or already-scheduled state', () => {
    expect(
      getAutosaveScheduleDecision({
        failedPersistedRevision: null,
        lastSavedPersistedRevision: 1,
        persistedRevision: 1,
        scheduledPersistedRevision: null,
      })
    ).toEqual({ failedPersistedRevision: null, shouldSchedule: false });

    expect(
      getAutosaveScheduleDecision({
        failedPersistedRevision: null,
        lastSavedPersistedRevision: 1,
        persistedRevision: 2,
        scheduledPersistedRevision: 2,
      })
    ).toEqual({ failedPersistedRevision: null, shouldSchedule: false });
  });

  it('holds a failed state key until a new durable edit happens', () => {
    expect(
      getAutosaveScheduleDecision({
        failedPersistedRevision: 2,
        lastSavedPersistedRevision: 1,
        persistedRevision: 2,
        scheduledPersistedRevision: null,
      })
    ).toEqual({ failedPersistedRevision: 2, shouldSchedule: false });

    expect(
      getAutosaveScheduleDecision({
        failedPersistedRevision: 2,
        lastSavedPersistedRevision: 1,
        persistedRevision: 3,
        scheduledPersistedRevision: null,
      })
    ).toEqual({ failedPersistedRevision: null, shouldSchedule: true });
  });

  it('clears a failed state key after returning to the last saved state', () => {
    expect(
      getAutosaveScheduleDecision({
        failedPersistedRevision: 2,
        lastSavedPersistedRevision: 1,
        persistedRevision: 1,
        scheduledPersistedRevision: null,
      })
    ).toEqual({ failedPersistedRevision: null, shouldSchedule: false });

    expect(
      getAutosaveScheduleDecision({
        failedPersistedRevision: null,
        lastSavedPersistedRevision: 1,
        persistedRevision: 2,
        scheduledPersistedRevision: null,
      })
    ).toEqual({ failedPersistedRevision: null, shouldSchedule: true });
  });
});

describe('isAutosaveCompletionCurrent', () => {
  it('rejects stale save completions after a newer durable revision exists', () => {
    expect(
      isAutosaveCompletionCurrent({
        completedPersistedRevision: 2,
        completedStateKey: 'state-2',
        currentPersistedRevision: 3,
        currentStateKey: 'state-3',
      })
    ).toBe(false);
  });

  it('accepts save completions for the current durable revision and state key', () => {
    expect(
      isAutosaveCompletionCurrent({
        completedPersistedRevision: 2,
        completedStateKey: 'state-2',
        currentPersistedRevision: 2,
        currentStateKey: 'state-2',
      })
    ).toBe(true);
  });
});
