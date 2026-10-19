import type { ModelConfig } from '@features/models/core/types';

import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  applyModelDraftFields,
  beginModelDraftSave,
  discardModelDraft,
  failModelDraftSave,
  finishModelDraftSave,
  getModelDraft,
  hasModelDraftConflict,
  hasUnsavedModelChanges,
  MAX_MODEL_DRAFTS,
  openModelIdentityDraft,
  recordModelDraftFields,
} from './modelDraftsStore';

const model = (key: string) => ({ key, name: key }) as ModelConfig;

describe('model drafts store', () => {
  beforeEach(() => {
    accountLifecycle.activate('model-drafts-a', ':user:model-drafts-a');
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('keeps only fields that differ from the server, each against the value it was first edited from', () => {
    const server = { description: '', name: 'Server' };

    recordModelDraftFields('m', 'identity', { description: '', name: 'Draft' }, server);
    // The server moved on before the next keystroke; the name keeps its original baseline.
    recordModelDraftFields('m', 'identity', { description: 'Notes', name: 'Draft 2' }, { ...server, name: 'Other' });

    expect(getModelDraft('m')?.identity).toEqual({
      baseline: { description: '', name: 'Server' },
      error: null,
      fields: { description: 'Notes', name: 'Draft 2' },
    });

    // Typing a field back to the server value stops it being an edit.
    recordModelDraftFields('m', 'identity', { description: '', name: 'Draft 2' }, server);
    expect(getModelDraft('m')?.identity?.fields).toEqual({ name: 'Draft 2' });
  });

  it('treats unset and null default settings as the same value', () => {
    recordModelDraftFields('m', 'defaults', { cfgScale: null, steps: 30 }, { steps: 30 });

    expect(getModelDraft('m')).toBeUndefined();
  });

  it('rebases untouched fields onto the server and flags only edited fields the server changed differently', () => {
    const draft = { baseline: { name: 'Server' }, fields: { name: 'Mine' } };

    expect(applyModelDraftFields({ description: 'New', name: 'Server 2' }, draft)).toEqual({
      description: 'New',
      name: 'Mine',
    });
    expect(hasModelDraftConflict({ description: 'New', name: 'Server' }, draft)).toBe(false);
    expect(hasModelDraftConflict({ description: '', name: 'Server 2' }, draft)).toBe(true);
    // The server arriving at the user's own value is agreement, not a conflict.
    expect(hasModelDraftConflict({ description: '', name: 'Mine' }, draft)).toBe(false);
  });

  it('keeps an open identity editor without edits, but only edits count as unsaved changes', () => {
    openModelIdentityDraft('m');
    expect(getModelDraft('m')?.identity).toBeDefined();
    expect(hasUnsavedModelChanges(getModelDraft('m'))).toBe(false);

    recordModelDraftFields('m', 'defaults', { steps: 20 }, { steps: 30 });
    expect(hasUnsavedModelChanges(getModelDraft('m'))).toBe(true);

    // Reverting the only defaults edit leaves the open editor behind.
    recordModelDraftFields('m', 'defaults', { steps: 30 }, { steps: 30 });
    expect(getModelDraft('m')).toEqual({ identity: { baseline: {}, error: null, fields: {} } });
  });

  it('discards one surface without touching the other, and drops the draft when nothing is left', () => {
    recordModelDraftFields('m', 'identity', { name: 'Mine' }, { name: 'Server' });
    beginModelDraftSave('m', 'identity', { name: 'Mine' });
    failModelDraftSave('m', 'identity', { name: 'Server' }, 'Name already taken.');
    recordModelDraftFields('m', 'defaults', { steps: 20 }, { steps: 30 });

    discardModelDraft('m', 'identity');
    expect(getModelDraft('m')).toEqual({ defaults: { baseline: { steps: 30 }, fields: { steps: 20 } } });

    discardModelDraft('m', 'defaults');
    expect(getModelDraft('m')).toBeUndefined();
  });

  it('evicts the least recently edited draft past the bound', () => {
    for (let index = 0; index < MAX_MODEL_DRAFTS; index += 1) {
      recordModelDraftFields(`m${index}`, 'defaults', { steps: index }, { steps: -1 });
    }

    // Editing the oldest again makes m1 the least recent.
    recordModelDraftFields('m0', 'defaults', { steps: 99 }, { steps: -1 });
    recordModelDraftFields('newest', 'defaults', { steps: 1 }, { steps: -1 });

    expect(getModelDraft('m0')?.defaults?.fields).toEqual({ steps: 99 });
    expect(getModelDraft('m1')).toBeUndefined();
    expect(getModelDraft('m2')).toBeDefined();
    expect(getModelDraft('newest')).toBeDefined();
  });

  it('lets a second open editor never evict unsaved work, and bounds open editors on their own', () => {
    recordModelDraftFields('dirty', 'defaults', { steps: 20 }, { steps: 30 });

    for (let index = 0; index <= MAX_MODEL_DRAFTS; index += 1) {
      openModelIdentityDraft(`open${index}`);
    }

    expect(getModelDraft('dirty')?.defaults?.fields).toEqual({ steps: 20 });
    expect(getModelDraft('open0')).toBeUndefined();
    expect(getModelDraft(`open${MAX_MODEL_DRAFTS}`)?.identity).toBeDefined();
  });

  it('ranks drafts by their last edit, not by opening or saving them', () => {
    for (let index = 0; index < MAX_MODEL_DRAFTS; index += 1) {
      recordModelDraftFields(`m${index}`, 'defaults', { steps: index }, { steps: -1 });
    }

    // Neither opening m0's editor nor a failed save of it makes it recent.
    openModelIdentityDraft('m0');
    beginModelDraftSave('m0', 'defaults', { steps: 0 });
    failModelDraftSave('m0', 'defaults', { steps: -1 }, null);
    recordModelDraftFields('newest', 'defaults', { steps: 1 }, { steps: -1 });

    expect(getModelDraft('m0')).toBeUndefined();
    expect(getModelDraft('m1')).toBeDefined();
  });

  it('refuses a second save while one is in flight', () => {
    recordModelDraftFields('m', 'identity', { name: 'Mine' }, { name: 'Server' });

    expect(beginModelDraftSave('m', 'identity', { name: 'Mine' })).toBe(true);
    expect(beginModelDraftSave('m', 'identity', { name: 'Mine' })).toBe(false);
  });

  it('settles a save by dropping what it sent and keeping what was edited after the click', () => {
    const server = { description: 'Old', name: 'Old' };

    recordModelDraftFields('m', 'identity', { description: 'Old', name: 'Sent' }, server);
    beginModelDraftSave('m', 'identity', { description: 'Old', name: 'Sent' });
    // After the click: a new description, and the name typed back to its old server value.
    recordModelDraftFields('m', 'identity', { description: 'Later', name: 'Old' }, server);

    finishModelDraftSave('m', 'identity', { description: 'Old', name: 'Sent' });

    expect(getModelDraft('m')?.identity).toEqual({
      baseline: { description: 'Old', name: 'Sent' },
      error: null,
      fields: { description: 'Later', name: 'Old' },
    });
  });

  it('closes the identity editor when a save leaves nothing to save', () => {
    recordModelDraftFields('m', 'identity', { name: 'Sent' }, { name: 'Old' });
    beginModelDraftSave('m', 'identity', { name: 'Sent' });

    finishModelDraftSave('m', 'identity', { name: 'Sent' });

    expect(getModelDraft('m')).toBeUndefined();
  });

  it('does not revive a draft discarded or evicted while its save was in flight', () => {
    recordModelDraftFields('m', 'identity', { name: 'Sent' }, { name: 'Old' });
    beginModelDraftSave('m', 'identity', { name: 'Sent' });
    discardModelDraft('m', 'identity');

    failModelDraftSave('m', 'identity', { name: 'Old' }, 'Rejected.');
    finishModelDraftSave('m', 'identity', { name: 'Sent' });

    expect(getModelDraft('m')).toBeUndefined();
  });

  it('drops edits the server comes to agree with, and the whole draft when none remain', () => {
    const library = [model('m'), model('other')];

    setModelsSnapshotForTests({ models: library, status: 'loaded' });
    recordModelDraftFields('m', 'defaults', { cfg_scale: 6, steps: 20 }, {});
    recordModelDraftFields('other', 'defaults', { steps: 20 }, {});

    setModelsSnapshotForTests({
      models: [
        { ...library[0]!, default_settings: { steps: 20 } },
        { ...library[1]!, default_settings: { steps: 20 } },
      ],
      status: 'loaded',
    });

    expect(getModelDraft('m')?.defaults?.fields).toEqual({ cfg_scale: 6 });
    expect(getModelDraft('other')).toBeUndefined();
  });

  it('drops drafts of models that leave the loaded library and keeps the rest', () => {
    setModelsSnapshotForTests({ models: [model('kept'), model('gone')], status: 'loaded' });
    recordModelDraftFields('kept', 'defaults', { steps: 20 }, { steps: 30 });
    recordModelDraftFields('gone', 'defaults', { steps: 20 }, { steps: 30 });

    setModelsSnapshotForTests({ models: [model('kept')], status: 'loaded' });

    expect(getModelDraft('kept')).toBeDefined();
    expect(getModelDraft('gone')).toBeUndefined();
  });

  it('clears every draft on an account transition', () => {
    recordModelDraftFields('m', 'identity', { name: 'Mine' }, { name: 'Server' });

    accountLifecycle.activate('model-drafts-b', ':user:model-drafts-b');

    expect(getModelDraft('m')).toBeUndefined();
  });
});
