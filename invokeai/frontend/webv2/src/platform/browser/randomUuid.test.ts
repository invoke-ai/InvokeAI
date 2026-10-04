import { createUuid, createUuidWith, type UuidCryptoSource } from '@platform/browser/randomUuid';
import { describe, expect, it, vi } from 'vitest';

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

/** A `Crypto` as seen from an insecure context: random bytes, no `randomUUID`. */
const insecureContextCrypto = (
  fill: (bytes: Uint8Array) => void = (bytes) => crypto.getRandomValues(bytes as Uint8Array<ArrayBuffer>)
) => {
  const source: UuidCryptoSource = {
    getRandomValues: ((array: ArrayBufferView) => {
      fill(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
      return array;
    }) as Crypto['getRandomValues'],
  };
  return source;
};

describe('createUuidWith', () => {
  it('delegates to the native generator when the context provides one', () => {
    const randomUUID = vi.fn(
      (): `${string}-${string}-${string}-${string}-${string}` => '11111111-2222-4333-8444-555555555555'
    );

    expect(createUuidWith({ getRandomValues: crypto.getRandomValues.bind(crypto), randomUUID })).toBe(
      '11111111-2222-4333-8444-555555555555'
    );
    expect(randomUUID).toHaveBeenCalledOnce();
  });

  it('builds a well-formed version 4 UUID without the native generator', () => {
    const source = insecureContextCrypto();
    const seen = new Set<string>();

    for (let index = 0; index < 200; index += 1) {
      const uuid = createUuidWith(source);
      expect(uuid).toMatch(UUID_V4);
      seen.add(uuid);
    }

    expect(seen.size).toBe(200);
  });

  it('forces the version and variant bits regardless of the random bytes', () => {
    const allOnes = createUuidWith(insecureContextCrypto((bytes) => bytes.fill(0xff)));
    const allZeros = createUuidWith(insecureContextCrypto((bytes) => bytes.fill(0x00)));

    expect(allOnes).toBe('ffffffff-ffff-4fff-bfff-ffffffffffff');
    expect(allZeros).toBe('00000000-0000-4000-8000-000000000000');
  });
});

describe('createUuid', () => {
  it('produces a version 4 UUID from the ambient crypto object', () => {
    expect(createUuid()).toMatch(UUID_V4);
  });

  it('survives an ambient crypto object that lacks randomUUID', () => {
    const original = globalThis.crypto;
    vi.stubGlobal(
      'crypto',
      insecureContextCrypto((bytes) => original.getRandomValues(bytes as Uint8Array<ArrayBuffer>))
    );
    try {
      expect(createUuid()).toMatch(UUID_V4);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
