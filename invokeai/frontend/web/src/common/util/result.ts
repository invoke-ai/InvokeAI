/**
 * Represents a successful result.
 * @template T The type of the value.
 */
export class Ok<T> {
  readonly value: T;
  constructor(value: T) {
    this.value = value;
  }

  /**
   * Type guard to check if this result is an `Ok` result.
   * @returns {this is Ok<T>} `true` if the result is an `Ok` result, otherwise `false`.
   */
  isOk(): this is Ok<T> {
    return true;
  }

  /**
   * Type guard to check if this result is an `Err` result.
   * @returns {this is Err<never>} `true` if the result is an `Err` result, otherwise `false`.
   */
  isErr(): this is Err<never> {
    return false;
  }
}

/**
 * Represents a failed result.
 * @template E The type of the error.
 */
export class Err<E> {
  readonly error: E;
  constructor(error: E) {
    this.error = error;
  }

  /**
   * Type guard to check if this result is an `Ok` result.
   * @returns {this is Ok<never>} `true` if the result is an `Ok` result, otherwise `false`.
   */
  isOk(): this is Ok<never> {
    return false;
  }

  /**
   * Type guard to check if this result is an `Err` result.
   * @returns {this is Err<E>} `true` if the result is an `Err` result, otherwise `false`.
   */
  isErr(): this is Err<E> {
    return true;
  }
}

/**
 * A union type that represents either a successful result (`Ok`) or a failed result (`Err`).
 * @template T The type of the value in the `Ok` case.
 * @template E The type of the error in the `Err` case.
 */
export type Result<T, E = Error> = Ok<T> | Err<E>;

/**
 * Creates a successful result.
 * @template T The type of the value.
 * @param {T} value The value to wrap in an `Ok` result.
 * @returns {Ok<T>} The `Ok` result containing the value.
 */
export function OkResult<T>(value: T): Ok<T> {
  return new Ok(value);
}

/**
 * Creates a failed result.
 * @template E The type of the error.
 * @param {E} error The error to wrap in an `Err` result.
 * @returns {Err<E>} The `Err` result containing the error.
 */
export function ErrResult<E>(error: E): Err<E> {
  return new Err(error);
}

/**
 * Wraps a synchronous function in a try-catch block, returning a `Result`.
 * @template T The type of the value returned by the function.
 * @param {() => T} fn The function to execute.
 * @returns {Result<T>} An `Ok` result if the function succeeds, or an `Err` result if it throws an error.
 */
export function withResult<T>(fn: () => T): Result<T> {
  try {
    return new Ok(fn());
  } catch (error) {
    return new Err(error instanceof Error ? error : new Error(String(error)));
  }
}

/**
 * Wraps an asynchronous function in a try-catch block, returning a `Promise` of a `Result`.
 * @template T The type of the value returned by the function.
 * @param {() => Promise<T>} fn The asynchronous function to execute.
 * @returns {Promise<Result<T>>} A `Promise` resolving to an `Ok` result if the function succeeds, or an `Err` result if it throws an error.
 */
export async function withResultAsync<T>(fn: () => Promise<T>): Promise<Result<T>> {
  try {
    const result = await fn();
    return new Ok(result);
  } catch (error) {
    return new Err(error instanceof Error ? error : new Error(String(error)));
  }
}
