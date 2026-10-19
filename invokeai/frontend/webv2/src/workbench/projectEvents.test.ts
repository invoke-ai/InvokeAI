import { describe, expect, it } from 'vitest';

import type { ProjectEvent } from './projectEventContracts';

import { prependProjectEvent, PROJECT_EVENT_LIMIT } from './projectEvents';

const event = (index: number): ProjectEvent => ({
  createdAt: `2026-09-04T00:00:${String(index).padStart(2, '0')}.000Z`,
  id: `event-${index}`,
  summary: `Event ${index}`,
  type: 'layout-updated',
});

describe('project events', () => {
  it('retains the newest bounded session events', () => {
    let events: ProjectEvent[] = [];

    for (let index = 0; index < PROJECT_EVENT_LIMIT + 25; index += 1) {
      events = prependProjectEvent(events, event(index));
    }

    expect(events).toHaveLength(PROJECT_EVENT_LIMIT);
    expect(events[0]?.id).toBe(`event-${PROJECT_EVENT_LIMIT + 24}`);
    expect(events.at(-1)?.id).toBe('event-25');
  });
});
