import { describe, expect, it } from 'vitest';

import { buildQueueItemOrigin, buildUtilityQueueItemOrigin, isWebv2QueueItemOrigin } from './events';

describe('isWebv2QueueItemOrigin', () => {
  it('recognizes every origin this client builds and no other client', () => {
    // Gallery skips these completions; an unrecognized form would double-refresh, a false match would miss one.
    expect(isWebv2QueueItemOrigin(buildQueueItemOrigin('local-1'))).toBe(true);
    expect(isWebv2QueueItemOrigin(buildQueueItemOrigin('local-1', 'project-1'))).toBe(true);
    expect(isWebv2QueueItemOrigin(buildUtilityQueueItemOrigin('filter-1'))).toBe(true);

    expect(isWebv2QueueItemOrigin('canvas')).toBe(false);
    expect(isWebv2QueueItemOrigin('chatbot:webv2:1')).toBe(false);
    expect(isWebv2QueueItemOrigin(null)).toBe(false);
  });
});
