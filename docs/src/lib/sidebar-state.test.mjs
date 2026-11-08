import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { resolveOpen, visibleScrollTop } from './sidebar-state.mjs';

describe('resolveOpen', () => {
  it("prefers the reader's last choice to the sidebar's default", () => {
    assert.equal(resolveOpen({ rendered: false, containsCurrent: false, stored: true }), true);
    assert.equal(resolveOpen({ rendered: true, containsCurrent: false, stored: false }), false);
  });

  it('keeps the default for a section the reader never toggled', () => {
    assert.equal(resolveOpen({ rendered: true, containsCurrent: false, stored: undefined }), true);
    assert.equal(resolveOpen({ rendered: false, containsCurrent: false, stored: 'yes' }), false);
  });

  it('always opens the section holding the current page', () => {
    assert.equal(resolveOpen({ rendered: true, containsCurrent: true, stored: false }), true);
  });
});

describe('visibleScrollTop', () => {
  it('keeps the position of a visible sidebar', () => {
    assert.equal(visibleScrollTop({ clientHeight: 800, scrollTop: 240 }), 240);
  });

  it('keeps nothing while the sidebar is hidden, so a closed drawer does not reset the position', () => {
    assert.equal(visibleScrollTop({ clientHeight: 0, scrollTop: 0 }), undefined);
    assert.equal(visibleScrollTop(null), undefined);
  });
});
