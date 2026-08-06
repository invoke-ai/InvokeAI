import { describe, expect, it } from 'vitest';

import { InvkFormatError } from './format';
import { buildInvkManifest, parseInvkManifest, toInvkFileName } from './manifest';

const v2 = {
  appVersion: '7.0',
  contents: 'workbench-project',
  createdAt: '2026-08-04T00:00:00.000Z',
  name: 'My project',
  version: 2,
};

const expectReason = (data: unknown, reason: string): void => {
  try {
    parseInvkManifest(data);
    expect.unreachable('parseInvkManifest should have thrown');
  } catch (error) {
    expect(error).toBeInstanceOf(InvkFormatError);
    expect((error as InvkFormatError).reason).toBe(reason);
  }
};

describe('parseInvkManifest', () => {
  it('accepts a v2 manifest', () => {
    expect(parseInvkManifest(v2)).toEqual(v2);
  });

  it('keeps the optional cover and source project id', () => {
    expect(parseInvkManifest({ ...v2, cover: 'cover.webp', sourceProjectId: 'project-1' })).toMatchObject({
      cover: 'cover.webp',
      sourceProjectId: 'project-1',
    });
  });

  /** The whole reason the schema is a union: naming what a v1 archive is. */
  it('names a legacy canvas project instead of dumping schema issues', () => {
    expectReason(
      { appVersion: '6.9', createdAt: '2026-04-14T00:00:00.000Z', name: 'Canvas', version: 1 },
      'legacy-canvas-project'
    );
  });

  it('reports an unknown numeric version as unsupported', () => {
    expectReason({ ...v2, version: 4 }, 'unsupported-version');
  });

  it('rejects a manifest missing the v2 discriminator', () => {
    expectReason({ appVersion: '7.0', createdAt: '', name: 'x', version: 2 }, 'unsupported-version');
  });

  it('rejects junk as not a project', () => {
    expectReason(null, 'not-a-project');
    expectReason({}, 'not-a-project');
    expectReason('a string', 'not-a-project');
  });
});

describe('buildInvkManifest', () => {
  it('omits the optional keys rather than writing undefined into the JSON', () => {
    const manifest = buildInvkManifest({ appVersion: '7.0', createdAt: v2.createdAt, name: 'My project' });

    expect(Object.keys(manifest).sort()).toEqual(['appVersion', 'contents', 'createdAt', 'name', 'version']);
  });

  it('round-trips through parse', () => {
    const manifest = buildInvkManifest({
      appVersion: '7.0',
      cover: 'cover.webp',
      createdAt: v2.createdAt,
      name: 'My project',
      sourceProjectId: 'project-1',
    });

    expect(parseInvkManifest(JSON.parse(JSON.stringify(manifest)))).toEqual(manifest);
  });
});

describe('toInvkFileName', () => {
  it('replaces characters filesystems reject', () => {
    expect(toInvkFileName('a/b\\c:d*e?f"g<h>i|j')).toBe('a_b_c_d_e_f_g_h_i_j.invk');
  });

  it('keeps non-Latin names intact', () => {
    expect(toInvkFileName('プロジェクト')).toBe('プロジェクト.invk');
  });

  it('keeps spaces and hyphens', () => {
    expect(toInvkFileName('My cool - project')).toBe('My cool - project.invk');
  });

  it('strips leading and trailing dots', () => {
    expect(toInvkFileName('..hidden..')).toBe('hidden.invk');
  });

  it('falls back for a name that sanitizes to nothing', () => {
    expect(toInvkFileName('   ')).toBe('project.invk');
    expect(toInvkFileName('...')).toBe('project.invk');
  });
});
