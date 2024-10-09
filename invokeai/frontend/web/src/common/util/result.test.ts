import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, expect, it } from 'vitest';

import { Err, ErrResult, Ok, OkResult, withResult, withResultAsync } from './result';

const promiseify = <T>(fn: () => T): (() => Promise<T>) => {
  return () =>
    new Promise((resolve) => {
      resolve(fn());
    });
};

describe('Result Utility Functions', () => {
  it('OkResult() should create an Ok result', () => {
    const result = OkResult(42);
    expect(result).toBeInstanceOf(Ok);
    expect(result.isOk()).toBe(true);
    expect(result.isErr()).toBe(false);
    expect(result.value).toBe(42);
    assert<Equals<Ok<number>, typeof result>>(result);
  });

  it('ErrResult() should create an Err result', () => {
    const error = new Error('Something went wrong');
    const result = ErrResult(error);
    expect(result).toBeInstanceOf(Err);
    expect(result.isOk()).toBe(false);
    expect(result.isErr()).toBe(true);
    expect(result.error).toBe(error);
    assert<Equals<Err<Error>, typeof result>>(result);
  });

  it('withResult() should return Ok on success', () => {
    const fn = () => 42;
    const result = withResult(fn);
    expect(result.isOk()).toBe(true);
    if (result.isOk()) {
      expect(result.value).toBe(42);
    }
  });

  it('withResult() should return Err on exception', () => {
    const fn = () => {
      throw new Error('Failure');
    };
    const result = withResult(fn);
    expect(result.isErr()).toBe(true);
    if (result.isErr()) {
      expect(result.error.message).toBe('Failure');
    }
  });

  it('withResultAsync() should return Ok on success', async () => {
    const fn = promiseify(() => 42);
    const result = await withResultAsync(fn);
    expect(result.isOk()).toBe(true);
    if (result.isOk()) {
      expect(result.value).toBe(42);
    }
  });

  it('withResultAsync() should return Err on exception', async () => {
    const fn = promiseify(() => {
      throw new Error('Async failure');
    });
    const result = await withResultAsync(fn);
    expect(result.isErr()).toBe(true);
    if (result.isErr()) {
      expect(result.error.message).toBe('Async failure');
    }
  });
});
