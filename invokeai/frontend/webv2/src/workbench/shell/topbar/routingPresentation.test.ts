import { describe, expect, it } from 'vitest';

import { getRoutingRelationship } from './routingPresentation';

describe('routing preview relationship', () => {
  it.each([
    [{ destinationLocked: false, sourceLocked: false }, 'arrow'],
    [{ destinationLocked: false, sourceLocked: true }, 'link'],
    [{ destinationLocked: true, sourceLocked: false }, 'link'],
    [{ destinationLocked: true, sourceLocked: true }, 'link'],
  ] as const)('maps route locks to the preview relationship', (locks, expected) => {
    expect(getRoutingRelationship(locks)).toBe(expected);
  });
});
