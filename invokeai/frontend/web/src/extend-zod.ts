import { assert } from 'tsafe';
import { z } from 'zod';

assert(!Object.hasOwn(z.ZodType.prototype, 'is'));

z.ZodType.prototype.is = function (val: unknown): val is z.infer<typeof this> {
  return this.safeParse(val).success;
};
