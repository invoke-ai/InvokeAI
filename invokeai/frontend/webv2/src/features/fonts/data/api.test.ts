import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => transport);

const dto = {
  axes: [{ default: 400, hidden: false, label: 'Weight', maximum: 900, minimum: 100, tag: 'wght' }],
  byte_size: 128,
  content_hash: 'a'.repeat(64),
  family: 'Example Sans',
  filename: 'ExampleSans.ttf',
  id: 'font/example',
  instances: [{ coordinates: { wght: 400 }, name: 'Regular' }],
  label: 'Example Sans Regular',
  scope: 'private',
  source: 'uploaded',
  style: 'normal',
  url: '/api/v1/fonts/font%2Fexample/file',
  weight: 400,
};

describe('font transport', () => {
  beforeEach(() => {
    transport.apiFetch.mockReset();
    transport.apiFetchJson.mockReset();
  });

  it('maps the paginated catalog and sends the server query contract', async () => {
    transport.apiFetchJson.mockResolvedValue({ items: [dto], limit: 25, offset: 50, total: 75 });
    const { listFonts } = await import('./api');

    await expect(
      listFonts({ contentHash: 'a'.repeat(64), limit: 25, offset: 50, scope: 'private', search: '  Sans  ' })
    ).resolves.toEqual({
      items: [
        expect.objectContaining({
          axes: [{ default: 400, hidden: false, label: 'Weight', maximum: 900, minimum: 100, tag: 'wght' }],
          byteSize: 128,
          contentHash: 'a'.repeat(64),
          instances: [{ coordinates: { wght: 400 }, name: 'Regular' }],
        }),
      ],
      limit: 25,
      offset: 50,
      total: 75,
    });
    expect(transport.apiFetchJson).toHaveBeenCalledWith(
      '/api/v1/fonts?limit=25&offset=50&content_hash=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&search=Sans&scope=private',
      {
        signal: undefined,
      }
    );
  });

  it('pins variation coordinates and sends the JSON content hash', async () => {
    transport.apiFetch.mockResolvedValue(new Response(new Uint8Array([1, 2, 3])));
    const { downloadFont } = await import('./api');
    const signal = new AbortController().signal;

    await expect(
      downloadFont(
        {
          axes: { wdth: 90, wght: 650 },
          contentHash: 'b'.repeat(64),
          id: 'font/example',
        },
        signal
      )
    ).resolves.toEqual(new Uint8Array([1, 2, 3]));
    expect(transport.apiFetch).toHaveBeenCalledWith('/api/v1/fonts/font%2Fexample/instance', {
      body: JSON.stringify({ content_hash: 'b'.repeat(64), coordinates: { wdth: 90, wght: 650 } }),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
      signal,
    });
  });

  it('validates multipart files, uploads private scope, and deletes by opaque id', async () => {
    transport.apiFetchJson
      .mockResolvedValueOnce({ ...dto, filename: 'checked.otf' })
      .mockResolvedValueOnce({ created: true, font: dto });
    transport.apiFetch.mockResolvedValue(new Response(null, { status: 204 }));
    const { deleteFont, uploadFont, validateFont } = await import('./api');
    const file = new File(['font bytes'], 'checked.otf', { type: 'font/otf' });

    await expect(validateFont(file)).resolves.toMatchObject({
      contentHash: 'a'.repeat(64),
      filename: 'checked.otf',
    });
    await expect(uploadFont(file, 'private')).resolves.toMatchObject({ created: true, font: { id: dto.id } });
    await deleteFont('font/example');

    const validationInit = transport.apiFetchJson.mock.calls[0]?.[1] as RequestInit;
    expect(validationInit.method).toBe('POST');
    expect(validationInit.body).toBeInstanceOf(FormData);
    expect((validationInit.body as FormData).get('file')).toBe(file);
    const uploadInit = transport.apiFetchJson.mock.calls[1]?.[1] as RequestInit;
    expect((uploadInit.body as FormData).get('scope')).toBe('private');
    expect(transport.apiFetch).toHaveBeenCalledWith('/api/v1/fonts/font%2Fexample', {
      method: 'DELETE',
      signal: undefined,
    });
  });
});
