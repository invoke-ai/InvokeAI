import { describe, expect, it } from 'vitest';

import { getNodeSettingKey, resolveNodeSettings } from './nodeSettingOverrides';

const base = {
  nodeId: 'node-1',
  isCanvasOutputNode: false,
  isIntermediate: undefined,
  useCache: undefined,
  nodeSettingValues: null,
  isAdmin: true,
};

describe('resolveNodeSettings', () => {
  it('falls back to the values saved in the workflow', () => {
    expect(resolveNodeSettings({ ...base, isIntermediate: true, useCache: false })).toEqual({
      is_intermediate: true,
      use_cache: false,
    });
  });

  it('falls back to the backend defaults when the node carries no values', () => {
    expect(resolveNodeSettings(base)).toEqual({ is_intermediate: false, use_cache: true });
  });

  // `save_to_gallery` is the inverse of `is_intermediate`.
  it.each([
    [true, false],
    [false, true],
  ])('applies a save_to_gallery override of %s as is_intermediate %s', (saveToGallery, isIntermediate) => {
    const nodeSettingValues = { [getNodeSettingKey('node-1', 'save_to_gallery')]: saveToGallery };
    expect(resolveNodeSettings({ ...base, isIntermediate: !isIntermediate, nodeSettingValues })).toEqual({
      is_intermediate: isIntermediate,
      use_cache: true,
    });
  });

  it('applies a use_cache override for an admin', () => {
    const nodeSettingValues = { [getNodeSettingKey('node-1', 'use_cache')]: false };
    expect(resolveNodeSettings({ ...base, useCache: true, nodeSettingValues })).toEqual({
      is_intermediate: false,
      use_cache: false,
    });
  });

  // The slice is persisted per browser, not per user, so an admin's leftover value must not be replayed.
  it('ignores a use_cache override for a non-admin', () => {
    const nodeSettingValues = { [getNodeSettingKey('node-1', 'use_cache')]: false };
    expect(resolveNodeSettings({ ...base, useCache: true, nodeSettingValues, isAdmin: false })).toEqual({
      is_intermediate: false,
      use_cache: true,
    });
  });

  it('ignores overrides belonging to another node', () => {
    const nodeSettingValues = {
      [getNodeSettingKey('other-node', 'save_to_gallery')]: true,
      [getNodeSettingKey('other-node', 'use_cache')]: false,
    };
    expect(resolveNodeSettings({ ...base, nodeSettingValues })).toEqual({ is_intermediate: false, use_cache: true });
  });

  // Canvas results land in the staging area; a gallery save there would produce a duplicate the canvas cannot stage.
  it('keeps the canvas output node intermediate regardless of any override', () => {
    const nodeSettingValues = { [getNodeSettingKey('node-1', 'save_to_gallery')]: true };
    expect(
      resolveNodeSettings({ ...base, isCanvasOutputNode: true, isIntermediate: false, nodeSettingValues })
    ).toEqual({ is_intermediate: true, use_cache: true });
  });
});
