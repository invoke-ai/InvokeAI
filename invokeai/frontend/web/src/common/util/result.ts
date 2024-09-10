/**
 * Represents a successful result.
 * @template T The type of the value.
 */
export type OkResult<T> = { type: 'Ok'; value: T };

/**
 * Represents a failed result.
 * @template E The type of the error.
 */
export type ErrResult<E> = { type: 'Err'; error: E };

/**
 * A union type that represents either a successful result (`Ok`) or a failed result (`Err`).
 * @template T The type of the value in the `Ok` case.
 * @template E The type of the error in the `Err` case.
 */
export type Result<T, E = Error> = OkResult<T> | ErrResult<E>;

/**
 * Creates a successful result.
 * @template T The type of the value.
 * @param {T} value The value to wrap in an `Ok` result.
 * @returns {OkResult<T>} The `Ok` result containing the value.
 */
export function Ok<T>(value: T): OkResult<T> {
  return { type: 'Ok', value };
}

/**
 * Creates a failed result.
 * @template E The type of the error.
 * @param {E} error The error to wrap in an `Err` result.
 * @returns {ErrResult<E>} The `Err` result containing the error.
 */
export function Err<E>(error: E): ErrResult<E> {
  return { type: 'Err', error };
}

/**
 * Wraps a synchronous function in a try-catch block, returning a `Result`.
 * @template T The type of the value returned by the function.
 * @param {() => T} fn The function to execute.
 * @returns {Result<T>} An `Ok` result if the function succeeds, or an `Err` result if it throws an error.
 */
export function withResult<T>(fn: () => T): Result<T> {
  try {
    return Ok(fn());
  } catch (error) {
    return Err(error instanceof Error ? error : new Error(String(error)));
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
    return Ok(result);
  } catch (error) {
    return Err(error instanceof Error ? error : new Error(String(error)));
  }
}

/**
 * Type guard to check if a `Result` is an `Ok` result.
 * @template T The type of the value in the `Ok` result.
 * @template E The type of the error in the `Err` result.
 * @param {Result<T, E>} result The result to check.
 * @returns {result is OkResult<T>} `true` if the result is an `Ok` result, otherwise `false`.
 */
export function isOk<T, E>(result: Result<T, E>): result is OkResult<T> {
  return result.type === 'Ok';
}

/**
 * Type guard to check if a `Result` is an `Err` result.
 * @template T The type of the value in the `Ok` result.
 * @template E The type of the error in the `Err` result.
 * @param {Result<T, E>} result The result to check.
 * @returns {result is ErrResult<E>} `true` if the result is an `Err` result, otherwise `false`.
 */
export function isErr<T, E>(result: Result<T, E>): result is ErrResult<E> {
  return result.type === 'Err';
}
