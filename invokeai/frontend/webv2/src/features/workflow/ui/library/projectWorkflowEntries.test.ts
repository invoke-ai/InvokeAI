import type { AuthSession } from '@features/identity';
import type { ProjectWorkflowEntry, ProjectWorkflowSource } from '@features/workflow/core/types';
import type { ProjectWorkflowOrder } from '@features/workflow/ui/workflowUiStore';

import { createProjectGraph } from '@features/workflow/utility';
import { describe, expect, it } from 'vitest';

import {
  countProjectWorkflowTags,
  isUpdatableSource,
  queryProjectWorkflowEntries,
  toProjectWorkflowLibraryEntries,
} from './projectWorkflowEntries';

const session = (overrides: Partial<AuthSession> = {}): AuthSession => ({
  accountEpoch: 1,
  expiredUserId: null,
  multiuserEnabled: true,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  switchingAccount: false,
  user: {
    created_at: '',
    display_name: null,
    email: '',
    is_active: true,
    is_admin: false,
    last_login_at: null,
    updated_at: '',
    user_id: 'user-a',
  },
  ...overrides,
});

const admin = session({
  user: {
    created_at: '',
    display_name: null,
    email: '',
    is_active: true,
    is_admin: true,
    last_login_at: null,
    updated_at: '',
    user_id: 'admin',
  },
});

const source = (overrides: Partial<ProjectWorkflowSource> = {}): ProjectWorkflowSource => ({
  libraryWorkflowId: 'library-1',
  ownerUserId: 'user-a',
  revision: 1,
  ...overrides,
});

describe('isUpdatableSource', () => {
  it('allows admins to update foreign saved records while keeping bundled, ordinary foreign, and unresolved sources blocked', () => {
    const foreignSource = source({ ownerUserId: 'user-b' });

    expect(isUpdatableSource(foreignSource, admin)).toBe(true);
    expect(isUpdatableSource(foreignSource, session())).toBe(false);
    expect(isUpdatableSource(source({ libraryWorkflowId: 'default_template' }), admin)).toBe(false);
    expect(isUpdatableSource(foreignSource, session({ ...admin, phase: 'unknown' }))).toBe(false);
  });
});

const workflow = (
  id: string,
  fields: { description?: string; lastRun?: string; name: string; tags?: string; updatedAt: string }
): ProjectWorkflowEntry => ({
  document: {
    ...createProjectGraph(id),
    description: fields.description ?? '',
    name: fields.name,
    tags: fields.tags ?? '',
    updatedAt: fields.updatedAt,
  },
  ...(fields.lastRun
    ? { lastRun: { completedAt: fields.lastRun, imageName: `${id}.png`, submittedAt: fields.lastRun } }
    : {}),
});

/** Project order: Upscale, portrait, Background, Sketch. */
const PROJECT = [
  workflow('upscale', { name: 'Upscale', tags: 'upscale, SDXL', updatedAt: '2026-11-03T00:00:00Z' }),
  workflow('portrait', {
    description: 'Soft studio light',
    lastRun: '2026-11-05T00:00:00Z',
    name: 'portrait',
    tags: 'sdxl',
    updatedAt: '2026-11-01T00:00:00Z',
  }),
  workflow('background', {
    lastRun: '2026-11-08T00:00:00Z',
    name: 'Background',
    tags: 'flux',
    updatedAt: '2026-11-02T00:00:00Z',
  }),
  workflow('sketch', { name: 'Sketch', updatedAt: '2026-11-04T00:00:00Z' }),
];
const ENTRIES = toProjectWorkflowLibraryEntries(PROJECT, null);
const ids = (search: string, tags: readonly string[], order: ProjectWorkflowOrder) =>
  queryProjectWorkflowEntries(ENTRIES, { order, search, tags }).map((entry) => entry.item.workflow_id);
const PROJECT_ORDER: ProjectWorkflowOrder = { direction: 'asc', sort: 'order' };

describe('queryProjectWorkflowEntries', () => {
  it('matches the search in names, descriptions and tags, and any checked tag, ignoring case', () => {
    expect(ids('STUDIO', [], PROJECT_ORDER)).toEqual(['portrait']);
    // A tag matches the search within it, as the server's search does.
    expect(ids('flu', [], PROJECT_ORDER)).toEqual(['background']);
    expect(ids('  ground ', [], PROJECT_ORDER)).toEqual(['background']);
    expect(ids('', ['sdxl', 'Flux'], PROJECT_ORDER)).toEqual(['upscale', 'portrait', 'background']);
    expect(ids('up', ['sdxl'], PROJECT_ORDER)).toEqual(['upscale']);
    // A tag is a whole token here, not a substring.
    expect(ids('', ['sd'], PROJECT_ORDER)).toEqual([]);
  });

  it('orders by project order, name, last update or last run in either direction', () => {
    expect(ids('', [], PROJECT_ORDER)).toEqual(['upscale', 'portrait', 'background', 'sketch']);
    expect(ids('', [], { direction: 'desc', sort: 'order' })).toEqual(['sketch', 'background', 'portrait', 'upscale']);
    expect(ids('', [], { direction: 'asc', sort: 'name' })).toEqual(['background', 'portrait', 'sketch', 'upscale']);
    expect(ids('', [], { direction: 'desc', sort: 'updated' })).toEqual([
      'sketch',
      'upscale',
      'background',
      'portrait',
    ]);
    // Never-run workflows come last either way, as the server orders them, in project order among themselves.
    expect(ids('', [], { direction: 'desc', sort: 'lastRun' })).toEqual([
      'background',
      'portrait',
      'upscale',
      'sketch',
    ]);
    expect(ids('', [], { direction: 'asc', sort: 'lastRun' })).toEqual(['portrait', 'background', 'upscale', 'sketch']);
  });
});

describe('countProjectWorkflowTags', () => {
  it('counts each workflow once per tag, merging case variants under the first casing, most used first', () => {
    expect(countProjectWorkflowTags(PROJECT)).toEqual([
      { count: 2, tag: 'SDXL' },
      { count: 1, tag: 'flux' },
      { count: 1, tag: 'upscale' },
    ]);
  });
});
