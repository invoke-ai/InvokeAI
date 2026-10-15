/** Generate UUIDv4 with randomUUID when available, otherwise getRandomValues. */

/** The subset of `Crypto` the generator relies on, so tests can swap it. */
export type UuidCryptoSource = Pick<Crypto, 'getRandomValues'> & Partial<Pick<Crypto, 'randomUUID'>>;

const toHex = (byte: number): string => byte.toString(16).padStart(2, '0');

export const createUuidWith = (source: UuidCryptoSource): string => {
  if (typeof source.randomUUID === 'function') {
    return source.randomUUID();
  }

  const bytes = new Uint8Array(16);
  source.getRandomValues(bytes);
  // Version 4: the top nibble of byte 6 is 0b0100.
  bytes[6] = (bytes[6]! & 0x0f) | 0x40;
  // RFC 4122 variant: the top two bits of byte 8 are 0b10.
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;

  const hex = Array.from(bytes, toHex).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
};

export const createUuid = (): string => createUuidWith(globalThis.crypto);
