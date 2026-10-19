import { ApiError } from '@platform/transport/http';
import { describe, expect, it } from 'vitest';

import { describeError, normalizeContext, normalizeError, serializeError, stripUrlSecrets } from './normalize';

describe('normalizeContext', () => {
  it('produces JSON-safe values for cycles, bigints, functions, symbols and dates', () => {
    const cyclic: Record<string, unknown> = { count: 2n, when: new Date('2026-09-22T10:00:00.000Z') };

    cyclic.self = cyclic;
    cyclic.run = function run() {
      return undefined;
    };
    cyclic.tag = Symbol('tag');
    cyclic.nan = Number.NaN;

    expect(normalizeContext(cyclic)).toEqual({
      truncated: false,
      value: {
        count: '2n',
        nan: 'NaN',
        run: '[function run]',
        self: '[circular]',
        tag: 'Symbol(tag)',
        when: '2026-09-22T10:00:00.000Z',
      },
    });
  });

  it('does not execute getters or toJSON and survives throwing accessors', () => {
    let getterCalls = 0;
    const value = {
      get computedGetter() {
        getterCalls += 1;

        return 'computed';
      },
      plain: 'kept',
      toJSON: () => {
        throw new Error('toJSON must not run');
      },
    };

    Object.defineProperty(value, 'explosive', {
      enumerable: true,
      get() {
        throw new Error('boom');
      },
    });

    expect(normalizeContext(value).value).toEqual({
      computedGetter: '[accessor]',
      explosive: '[accessor]',
      plain: 'kept',
      toJSON: '[function toJSON]',
    });
    expect(getterCalls).toBe(0);
  });

  it('redacts credential-like keys and strips URL queries and fragments', () => {
    expect(
      normalizeContext({
        Authorization: 'Bearer abc',
        apiKey: 'k',
        nested: { password: 'p', refreshToken: 't', sessionCookie: 'c' },
        path: '/api/v1/images?token=abc#frag',
        url: 'https://host.example/api/v1/x?access_token=abc#x',
        username: 'visible',
      }).value
    ).toEqual({
      Authorization: '[redacted]',
      apiKey: '[redacted]',
      nested: { password: '[redacted]', refreshToken: '[redacted]', sessionCookie: '[redacted]' },
      path: '/api/v1/images',
      url: 'https://host.example/api/v1/x',
      username: 'visible',
    });
    expect(stripUrlSecrets('plain text with ? mark')).toBe('plain text with ? mark');
  });

  it('bounds depth, collection sizes and string lengths with explicit markers', () => {
    const deep = { l1: { l2: { l3: { l4: { l5: { l6: { l7: 'too deep' } } } } } } };
    const wide = Array.from({ length: 60 }, (_, index) => index);
    const long = 'x'.repeat(1_200);
    const result = normalizeContext({ deep, long, wide });

    expect(result.truncated).toBe(true);
    expect(result.value).toMatchObject({
      deep: { l1: { l2: { l3: { l4: { l5: '[depth limit]' } } } } },
      long: `${'x'.repeat(1_000)}…[+200 chars]`,
    });
    expect((result.value as { wide: unknown[] }).wide).toHaveLength(51);
    expect((result.value as { wide: unknown[] }).wide[50]).toBe('[+10 more]');
  });

  it('detects cycles that pass through an error and keeps status reads to own properties', () => {
    const error = new Error('root') as Error & { cause?: unknown };
    let getterCalls = 0;

    class StatusError extends Error {
      get status(): number {
        getterCalls += 1;

        return 500;
      }
    }

    error.cause = { via: error };

    const result = normalizeContext({ error });

    expect(result.value).toEqual({
      error: { cause: { via: '[circular]' }, message: 'root', name: 'Error', stack: expect.any(String) },
    });
    expect(normalizeError(new StatusError('boom')).error).not.toHaveProperty('status');
    expect(getterCalls).toBe(0);
  });

  it('redacts credential-like map keys', () => {
    expect(
      normalizeContext(
        new Map([
          ['authorization', 'Bearer x'],
          ['plain', 'kept'],
        ])
      ).value
    ).toEqual([
      ['authorization', '[redacted]'],
      ['plain', 'kept'],
    ]);
  });

  it('describes binary payloads and maps instead of copying them', () => {
    const result = normalizeContext({
      buffer: new Uint8Array(16),
      map: new Map([['a', 1]]),
      set: new Set(['b']),
    });

    expect(result.value).toEqual({ buffer: '[binary 16 bytes]', map: [['a', 1]], set: ['b'] });
  });
});

describe('normalizeError', () => {
  it('captures name, message, stack, cause chain and API status without retaining the instance', () => {
    const cause = new Error('root cause');
    const error = new ApiError('{"detail":"bad"}', 502, undefined);

    (error as Error & { cause?: unknown }).cause = cause;

    const { error: serialized, truncated } = normalizeError(error);

    expect(truncated).toBe(false);
    expect(serialized).toMatchObject({ message: '{"detail":"bad"}', name: 'ApiError', status: 502 });
    expect(serialized.stack).toContain('ApiError');
    expect(serialized.cause).toMatchObject({ message: 'root cause', name: 'Error' });
    expect(serialized).not.toHaveProperty('headers');
  });

  it('truncates very long stacks and cuts cause chains after three levels', () => {
    const error = new Error('top');
    let current: Error & { cause?: unknown } = error;

    for (let depth = 0; depth < 5; depth += 1) {
      const next = new Error(`cause ${depth}`);

      current.cause = next;
      current = next;
    }
    error.stack = 'x'.repeat(5_000);

    const { error: serialized, truncated } = normalizeError(error);

    expect(truncated).toBe(true);
    expect(serialized.stack).toHaveLength(4_001);
    expect(JSON.stringify(serialized)).toContain('[cause depth limit]');
  });

  it('serializes non-error throwables', () => {
    expect(serializeError('plain string')).toEqual({ message: 'plain string', name: 'NonError' });
    expect(serializeError({ code: 'E1' })).toEqual({ message: '{"code":"E1"}', name: 'NonError' });
  });
});

describe('describeError', () => {
  it('never renders [object Object]', () => {
    expect(describeError(new Error('boom'))).toBe('boom');
    expect(describeError({ message: 'from object' })).toBe('from object');
    expect(describeError({ status: 500 })).toBe('{"status":500}');
    expect(describeError(undefined)).toBe('');
    expect(describeError(42)).toBe('42');
  });
});
