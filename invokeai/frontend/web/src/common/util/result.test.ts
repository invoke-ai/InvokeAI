import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { describe, expect, it } from 'vitest';

import type { ErrResult, OkResult } from './result';
import { Err, isErr, isOk, Ok, withResult, withResultAsync } from './result'; // Adjust import as needed

const promiseify = <T>(fn: () => T): (() => Promise<T>) => {
  return () =>
    new Promise((resolve) => {
      resolve(fn());
    });
};

describe('Result Utility Functions', () => {
  it('Ok() should create an OkResult', () => {
    const result = Ok(42);
    expect(result).toEqual({ type: 'Ok', value: 42 });
    expect(isOk(result)).toBe(true);
    expect(isErr(result)).toBe(false);
    assert<Equals<OkResult<number>, typeof result>>(result);
  });

  it('Err() should create an ErrResult', () => {
    const error = new Error('Something went wrong');
    const result = Err(error);
    expect(result).toEqual({ type: 'Err', error });
    expect(isOk(result)).toBe(false);
    expect(isErr(result)).toBe(true);
    assert<Equals<ErrResult<Error>, typeof result>>(result);
  });

  it('withResult() should return Ok on success', () => {
    const fn = () => 42;
    const result = withResult(fn);
    expect(isOk(result)).toBe(true);
    if (isOk(result)) {
      expect(result.value).toBe(42);
    }
  });

  it('withResult() should return Err on exception', () => {
    const fn = () => {
      throw new Error('Failure');
    };
    const result = withResult(fn);
    expect(isErr(result)).toBe(true);
    if (isErr(result)) {
      expect(result.error.message).toBe('Failure');
    }
  });

  it('withResultAsync() should return Ok on success', async () => {
    const fn = promiseify(() => 42);
    const result = await withResultAsync(fn);
    expect(isOk(result)).toBe(true);
    if (isOk(result)) {
      expect(result.value).toBe(42);
    }
  });

  it('withResultAsync() should return Err on exception', async () => {
    const fn = promiseify(() => {
      throw new Error('Async failure');
    });
    const result = await withResultAsync(fn);
    expect(isErr(result)).toBe(true);
    if (isErr(result)) {
      expect(result.error.message).toBe('Async failure');
    }
  });
});
