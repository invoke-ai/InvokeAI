import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createFontArchiveTransport } from './fontTransport';
import { FontImportQuotaError } from './format';

const fonts = vi.hoisted(() => ({
  deleteFont: vi.fn(),
  downloadFont: vi.fn(),
  getFont: vi.fn(),
  listFonts: vi.fn(),
  uploadFont: vi.fn(),
  validateFont: vi.fn(),
}));
vi.mock('@features/fonts', () => fonts);

const dependency = {
  contentHash: 'a'.repeat(64),
  family: 'Example',
  label: 'Example Regular',
  references: ['old-server-id'],
};

describe('archive font transport', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    accountLifecycle.activate('font-import-owner');
  });

  it('resolves a cross-server id collision by exact content instead of downloading the wrong face', async () => {
    fonts.getFont.mockResolvedValue({ contentHash: 'b'.repeat(64), id: 'old-server-id' });
    fonts.listFonts.mockResolvedValue({
      items: [{ contentHash: dependency.contentHash, filename: 'original.ttf', id: 'local-id' }],
    });
    fonts.downloadFont.mockResolvedValue(new Uint8Array([1, 2]));

    await expect(createFontArchiveTransport().download(dependency)).resolves.toEqual({
      bytes: new Uint8Array([1, 2]),
      filename: 'original.ttf',
    });
    expect(fonts.listFonts).toHaveBeenCalledWith(
      { contentHash: dependency.contentHash, limit: 1 },
      expect.any(AbortSignal)
    );
    expect(fonts.downloadFont).toHaveBeenCalledWith(
      { contentHash: dependency.contentHash, id: 'local-id' },
      expect.any(AbortSignal)
    );
  });

  it.each(['validate', 'upload'] as const)(
    'offers references-only recovery for a server limit during %s',
    async (method) => {
      fonts.validateFont.mockRejectedValue(new ApiError('File exceeds configured limit', 413));
      fonts.uploadFont.mockRejectedValue(new ApiError('Library is full', 413));

      await expect(createFontArchiveTransport()[method](new File(['font'], 'font.ttf'))).rejects.toBeInstanceOf(
        FontImportQuotaError
      );
    }
  );

  it('rejects work owned by the previous account before making requests', async () => {
    const transport = createFontArchiveTransport();
    accountLifecycle.activate('another-owner');

    await expect(transport.download(dependency)).rejects.toThrow();
    await expect(transport.upload(new File(['font'], 'font.ttf'))).rejects.toThrow();
    expect(fonts.getFont).not.toHaveBeenCalled();
    expect(fonts.uploadFont).not.toHaveBeenCalled();
  });
});
