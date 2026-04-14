import { omit, pick } from 'es-toolkit/compat';
import { schema, templates } from 'features/nodes/store/util/testUtils';
import { parseSchema } from 'features/nodes/util/schema/parseSchema';
import { describe, expect, it } from 'vitest';

const stripUndefinedDeep = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

describe('parseSchema', () => {
  it('should parse the schema', () => {
    const parsed = parseSchema(schema);
    expect(stripUndefinedDeep(parsed)).toEqual(stripUndefinedDeep(templates));
  });
  it('should omit denied nodes', () => {
    const parsed = parseSchema(schema, undefined, ['add']);
    expect(stripUndefinedDeep(parsed)).toEqual(stripUndefinedDeep(omit(templates, 'add')));
  });
  it('should include only allowed nodes', () => {
    const parsed = parseSchema(schema, ['add']);
    expect(stripUndefinedDeep(parsed)).toEqual(stripUndefinedDeep(pick(templates, 'add')));
  });
});
