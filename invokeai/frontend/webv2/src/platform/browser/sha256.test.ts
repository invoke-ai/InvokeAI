import { sha256, sha256Fallback, sha256Hex } from '@platform/browser/sha256';
import { describe, expect, it, vi } from 'vitest';

const encode = (text: string): Uint8Array => new TextEncoder().encode(text);
const hex = (bytes: Uint8Array): string => Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');

const nativeHex = async (bytes: Uint8Array): Promise<string> =>
  hex(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>)));

describe('sha256Fallback', () => {
  it('matches the published test vectors', () => {
    expect(hex(sha256Fallback(encode('')))).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    expect(hex(sha256Fallback(encode('abc')))).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    expect(hex(sha256Fallback(encode('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq')))).toBe(
      '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1'
    );
  });

  it('agrees with Web Crypto across the padding boundaries', async () => {
    for (const length of [0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 128, 1000, 70_000]) {
      const bytes = new Uint8Array(length);
      for (let index = 0; index < length; index += 1) {
        bytes[index] = (index * 31 + length) & 0xff;
      }
      expect(hex(sha256Fallback(bytes)), `length ${length}`).toBe(await nativeHex(bytes));
    }
  });

  it('hashes a subarray by its own bytes, not the whole backing buffer', async () => {
    const backing = new Uint8Array(64).fill(0xab);
    const window = backing.subarray(8, 24);

    expect(hex(sha256Fallback(window))).toBe(await nativeHex(new Uint8Array(window)));
  });
});

describe('sha256 / sha256Hex', () => {
  it('uses the native digest when the context exposes one', async () => {
    const digest = vi.fn(crypto.subtle.digest.bind(crypto.subtle));
    const subtle = { digest } as unknown as SubtleCrypto;

    expect(await sha256Hex(encode('abc'), subtle)).toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    );
    expect(digest).toHaveBeenCalledOnce();
  });

  it('falls back to the JavaScript digest without one, with identical output', async () => {
    const bytes = encode('The quick brown fox jumps over the lazy dog');

    expect(await sha256Hex(bytes, undefined)).toBe(await nativeHex(bytes));
    expect(hex(await sha256(bytes.buffer as ArrayBuffer, undefined))).toBe(await nativeHex(bytes));
  });

  it('survives an ambient crypto object that lacks subtle', async () => {
    const original = globalThis.crypto;
    vi.stubGlobal('crypto', { getRandomValues: original.getRandomValues.bind(original) });
    try {
      expect(await sha256Hex(encode('abc'))).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
