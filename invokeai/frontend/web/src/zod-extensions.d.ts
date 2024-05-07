import 'zod';

declare module 'zod' {
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  export interface ZodType<Output = any> {
    is(val: unknown): val is Output;
  }
}
