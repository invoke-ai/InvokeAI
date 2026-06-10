import type { S } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, expect, it, test } from 'vitest';

import type { BoardRecordOrderBy } from './types';
import { isCanvasProjectName, isVideoName } from './types';

describe('Gallery Types', () => {
  // Ensure zod types match OpenAPI types
  test('BoardRecordOrderBy', () => {
    assert<Equals<BoardRecordOrderBy, S['BoardRecordOrderBy']>>();
  });
});

// The discriminators below are what GalleryImageGrid, useGalleryItemDTO and the bulk-delete
// splitter dispatch on. The SimpleNameService emits images with `.png`, videos with `.mp4`, and
// canvas projects as bare UUIDs (no extension) — the discriminators have to stay mutually
// exclusive so a mixed-type selection routes each name to the correct API endpoint.

describe('isVideoName', () => {
  it('matches names ending in .mp4 (case-insensitive)', () => {
    expect(isVideoName('a.mp4')).toBe(true);
    expect(isVideoName('A.MP4')).toBe(true);
  });

  it('rejects image and project names', () => {
    expect(isVideoName('aaf9504b-f1a2-4410-bf43-96f700c49246.png')).toBe(false);
    expect(isVideoName('aaf9504b-f1a2-4410-bf43-96f700c49246')).toBe(false);
  });
});

describe('isCanvasProjectName', () => {
  it('matches bare UUID v4 strings', () => {
    expect(isCanvasProjectName('aaf9504b-f1a2-4410-bf43-96f700c49246')).toBe(true);
    expect(isCanvasProjectName('AAF9504B-F1A2-4410-BF43-96F700C49246')).toBe(true);
  });

  it('rejects names with an extension', () => {
    expect(isCanvasProjectName('aaf9504b-f1a2-4410-bf43-96f700c49246.png')).toBe(false);
    expect(isCanvasProjectName('aaf9504b-f1a2-4410-bf43-96f700c49246.mp4')).toBe(false);
    expect(isCanvasProjectName('aaf9504b-f1a2-4410-bf43-96f700c49246.invk')).toBe(false);
  });

  it('rejects malformed UUIDs', () => {
    expect(isCanvasProjectName('not-a-uuid')).toBe(false);
    expect(isCanvasProjectName('aaf9504b-f1a2-4410-bf43')).toBe(false);
    expect(isCanvasProjectName('')).toBe(false);
  });

  it('is mutually exclusive with isVideoName for any single name', () => {
    const names = [
      'aaf9504b-f1a2-4410-bf43-96f700c49246', // project
      'aaf9504b-f1a2-4410-bf43-96f700c49246.png', // image
      'aaf9504b-f1a2-4410-bf43-96f700c49246.mp4', // video
    ];
    for (const name of names) {
      const matches = [isCanvasProjectName(name), isVideoName(name)].filter(Boolean);
      // At most one of the two discriminators may fire for any name. (Images are the
      // "neither matches" fall-through bucket — they don't get their own discriminator
      // function.)
      expect(matches.length).toBeLessThanOrEqual(1);
    }
  });
});
