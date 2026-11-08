import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { splitSitePath } from './locales.mjs';

describe('splitSitePath', () => {
  it('separates the locale from the English path, without the base path', () => {
    assert.deepEqual(splitSitePath('/InvokeAI/de/users-guide/x/#y', '/InvokeAI'), {
      locale: 'de',
      path: '/users-guide/x/#y',
    });
    assert.deepEqual(splitSitePath('/hi', '/'), { locale: 'hi', path: '/' });
    assert.deepEqual(splitSitePath('/es#top', '/'), { locale: 'es', path: '/#top' });
    assert.deepEqual(splitSitePath('/InvokeAI', '/InvokeAI/'), { locale: undefined, path: '/' });
  });

  it('only treats a known locale as a locale segment', () => {
    assert.deepEqual(splitSitePath('/development/x/', '/'), { locale: undefined, path: '/development/x/' });
    assert.deepEqual(splitSitePath('/it/x/', '/'), { locale: undefined, path: '/it/x/' });
  });

  it('is undefined outside the base path', () => {
    assert.equal(splitSitePath('/InvokeAIx/page/', '/InvokeAI'), undefined);
    assert.equal(splitSitePath('/other/', '/InvokeAI'), undefined);
  });
});
