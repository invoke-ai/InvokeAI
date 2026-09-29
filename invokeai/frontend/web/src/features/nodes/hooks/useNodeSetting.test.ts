import { describe, expect, it } from 'vitest';

import { getIsNodeSettingPermitted } from './useNodeSetting';

// `use_cache` toggles the process-global invocation cache for a node, which is an admin-only policy. The form builder
// can carry a `use_cache` element into a shared workflow, so every surface that renders one - node footer, form view
// mode, form edit mode - must consult this predicate, not just the footer.
describe('getIsNodeSettingPermitted', () => {
  it('permits use_cache only for admins', () => {
    expect(getIsNodeSettingPermitted('use_cache', true)).toBe(true);
    expect(getIsNodeSettingPermitted('use_cache', false)).toBe(false);
  });

  it('permits save_to_gallery for everyone', () => {
    expect(getIsNodeSettingPermitted('save_to_gallery', true)).toBe(true);
    expect(getIsNodeSettingPermitted('save_to_gallery', false)).toBe(true);
  });
});
