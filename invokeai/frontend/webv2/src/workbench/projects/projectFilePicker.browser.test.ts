import { afterEach, describe, expect, it, vi } from 'vitest';

import { pickProjectFile } from './projectFile';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('pickProjectFile', () => {
  it('offers both current archives and shipped legacy JSON project files', async () => {
    let acceptedTypes = '';

    vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(function (this: HTMLInputElement) {
      acceptedTypes = this.accept;
      this.oncancel?.(new Event('cancel'));
    });

    await expect(pickProjectFile()).resolves.toBeNull();
    expect(acceptedTypes.split(',')).toEqual(['.invk', '.invokeproject.json']);
  });
});
