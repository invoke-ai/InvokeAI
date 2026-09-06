import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { afterEach, describe, expect, it, vi } from 'vitest';

import { openMediaInNewTab } from './openMediaInNewTab';

const readSource = (relativePath: string) =>
  readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8');

describe('openMediaInNewTab', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('opens the media URL directly in a new tab', () => {
    const open = vi.fn(() => null);
    vi.stubGlobal('window', { open });

    openMediaInNewTab('api/v1/images/i/test.png/full');

    expect(open).toHaveBeenCalledWith('api/v1/images/i/test.png/full', '_blank', 'noopener,noreferrer');
  });

  it('does not use an about:blank intermediary', () => {
    const source = readSource('./openMediaInNewTab.ts');

    expect(source).not.toContain('about:blank');
  });

  it('keeps middle-click opening on the shared direct opener', () => {
    const source = readSource('../hooks/useMiddleClickOpenInNewTab.ts');

    expect(source).toContain("from 'common/util/openMediaInNewTab'");
    expect(source).not.toContain('useMediaCookieRefresh');
  });
});
