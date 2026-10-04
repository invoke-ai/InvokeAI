import { describe, expect, it } from 'vitest';

import { createCanvasDiagnostics } from './diagnostics';

describe('createCanvasDiagnostics', () => {
  it('is a stable no-op when disabled', () => {
    const diagnostics = createCanvasDiagnostics();
    const before = diagnostics.snapshot();
    diagnostics.increment('surfaceCreations');
    diagnostics.add('allocatedBaseBytes', 400);
    expect(diagnostics.snapshot()).toBe(before);
    expect(before.surfaceCreations).toBe(0);
    expect(before.allocatedBaseBytes).toBe(0);
  });

  it('records deterministic counters when enabled and resets explicitly', () => {
    const diagnostics = createCanvasDiagnostics(true);
    diagnostics.increment('derivedCacheHits');
    diagnostics.increment('derivedCacheHits');
    diagnostics.add('allocatedDerivedBytes', 256);
    expect(diagnostics.snapshot()).toMatchObject({ allocatedDerivedBytes: 256, derivedCacheHits: 2 });
    diagnostics.reset();
    expect(diagnostics.snapshot()).toMatchObject({ allocatedDerivedBytes: 0, derivedCacheHits: 0 });
  });
});
