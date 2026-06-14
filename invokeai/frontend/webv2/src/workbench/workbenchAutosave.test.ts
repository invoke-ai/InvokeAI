import { describe, expect, it } from 'vitest';

import { getAutosaveScheduleDecision } from './workbenchAutosave';

describe('getAutosaveScheduleDecision', () => {
  it('does not schedule already-saved or already-scheduled state', () => {
    expect(
      getAutosaveScheduleDecision({
        failedStateKey: null,
        lastSavedStateKey: 'project-a',
        persistedStateKey: 'project-a',
        scheduledStateKey: null,
      })
    ).toEqual({ failedStateKey: null, shouldSchedule: false });

    expect(
      getAutosaveScheduleDecision({
        failedStateKey: null,
        lastSavedStateKey: 'project-a',
        persistedStateKey: 'project-b',
        scheduledStateKey: 'project-b',
      })
    ).toEqual({ failedStateKey: null, shouldSchedule: false });
  });

  it('holds a failed state key until a new durable edit happens', () => {
    expect(
      getAutosaveScheduleDecision({
        failedStateKey: 'project-b',
        lastSavedStateKey: 'project-a',
        persistedStateKey: 'project-b',
        scheduledStateKey: null,
      })
    ).toEqual({ failedStateKey: 'project-b', shouldSchedule: false });

    expect(
      getAutosaveScheduleDecision({
        failedStateKey: 'project-b',
        lastSavedStateKey: 'project-a',
        persistedStateKey: 'project-c',
        scheduledStateKey: null,
      })
    ).toEqual({ failedStateKey: null, shouldSchedule: true });
  });

  it('clears a failed state key after returning to the last saved state', () => {
    expect(
      getAutosaveScheduleDecision({
        failedStateKey: 'project-b',
        lastSavedStateKey: 'project-a',
        persistedStateKey: 'project-a',
        scheduledStateKey: null,
      })
    ).toEqual({ failedStateKey: null, shouldSchedule: false });

    expect(
      getAutosaveScheduleDecision({
        failedStateKey: null,
        lastSavedStateKey: 'project-a',
        persistedStateKey: 'project-b',
        scheduledStateKey: null,
      })
    ).toEqual({ failedStateKey: null, shouldSchedule: true });
  });
});
