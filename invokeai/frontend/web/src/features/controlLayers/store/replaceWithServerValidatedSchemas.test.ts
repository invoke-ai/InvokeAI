import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import { beforeEach, describe, expect, it } from 'vitest';
import { z, ZodError } from 'zod';

import {
  clearSchemaReplacements,
  registerSchemaReplacement,
  replaceWithServerValidatedSchemas,
} from './replaceWithServerValidatedSchemas';

describe('replaceWithServerValidatedSchemas', () => {
  beforeEach(() => {
    clearSchemaReplacements();
  });

  const zFoo = z.literal('foo');

  const zFooAsyncOK = zFoo.refine(() => {
    return Promise.resolve(true);
  });

  const zFooAsyncFAIL = zFoo.refine(() => {
    return Promise.resolve(false);
  });

  it('should should not alter the type of the schema', () => {
    const zTest = z.object({
      foo: zFoo,
    });
    registerSchemaReplacement(zFoo, zFooAsyncOK);
    const _serverValidatedSchema = replaceWithServerValidatedSchemas(zTest);

    assert<Equals<z.infer<typeof _serverValidatedSchema>, z.infer<typeof zTest>>>();
  });

  it('should pass validation when the replaced async validator passes', async () => {
    const zTest = z.object({
      foo: zFoo,
    });
    registerSchemaReplacement(zFoo, zFooAsyncOK);
    const serverValidatedSchema = replaceWithServerValidatedSchemas(zTest);

    expect(() => serverValidatedSchema.parse({ foo: 'foo' })).toThrow(
      'Encountered Promise during synchronous parse. Use .parseAsync() instead.'
    );
    await expect(serverValidatedSchema.parseAsync({ foo: 'foo' })).resolves.toEqual({ foo: 'foo' });
  });

  it('should fail validation when the replaced async validator fails', async () => {
    const zTest = z.object({
      foo: zFoo,
    });
    registerSchemaReplacement(zFoo, zFooAsyncFAIL);
    const serverValidatedSchema = replaceWithServerValidatedSchemas(zTest);

    expect(() => serverValidatedSchema.parse({ foo: 'foo' })).toThrow(
      'Encountered Promise during synchronous parse. Use .parseAsync() instead.'
    );
    await expect(serverValidatedSchema.parseAsync({ foo: 'foo' })).rejects.toThrow(ZodError);
  });

  it('should handle deeply-nested objects', async () => {
    const zNested = z.object({
      nested: z.object({
        foo: zFoo,
      }),
    });

    registerSchemaReplacement(zFoo, zFooAsyncOK);
    const serverValidatedSchema = replaceWithServerValidatedSchemas(zNested);

    expect(() => serverValidatedSchema.parse({ nested: { foo: 'foo' } })).toThrow(
      'Encountered Promise during synchronous parse. Use .parseAsync() instead.'
    );

    await expect(serverValidatedSchema.parseAsync({ nested: { foo: 'foo' } })).resolves.toEqual({
      nested: { foo: 'foo' },
    });
  });

  it('should handle arrays', async () => {
    const zArray = z.array(zFoo);

    registerSchemaReplacement(zFoo, zFooAsyncOK);
    const serverValidatedSchema = replaceWithServerValidatedSchemas(zArray);

    expect(() => serverValidatedSchema.parse(['foo', 'foo'])).toThrow(
      'Encountered Promise during synchronous parse. Use .parseAsync() instead.'
    );

    await expect(serverValidatedSchema.parseAsync(['foo', 'foo'])).resolves.toEqual(['foo', 'foo']);
  });

  it('should handle sets', async () => {
    const zSet = z.set(zFoo);

    registerSchemaReplacement(zFoo, zFooAsyncOK);
    const serverValidatedSchema = replaceWithServerValidatedSchemas(zSet);

    expect(() => serverValidatedSchema.parse(new Set(['foo', 'foo']))).toThrow(
      'Encountered Promise during synchronous parse. Use .parseAsync() instead.'
    );

    await expect(serverValidatedSchema.parseAsync(new Set(['foo', 'foo']))).resolves.toEqual(new Set(['foo']));
  });

  it('should handle records', async () => {
    const zRecord = z.record(z.string(), zFoo);

    registerSchemaReplacement(zFoo, zFooAsyncOK);
    const serverValidatedSchema = replaceWithServerValidatedSchemas(zRecord);

    expect(() => serverValidatedSchema.parse({ a: 'foo', b: 'foo' })).toThrow(
      'Encountered Promise during synchronous parse. Use .parseAsync() instead.'
    );

    await expect(serverValidatedSchema.parseAsync({ a: 'foo', b: 'foo' })).resolves.toEqual({ a: 'foo', b: 'foo' });
  });
});
