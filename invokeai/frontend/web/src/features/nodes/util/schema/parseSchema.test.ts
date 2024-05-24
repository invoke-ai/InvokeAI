import { schema, templates } from 'features/nodes/store/util/testUtils';
import { parseSchema } from 'features/nodes/util/schema/parseSchema';
import { omit, pick } from 'lodash-es';
import { describe, expect, it } from 'vitest';

describe('parseSchema', () => {
  it('should parse the schema', () => {
    const parsed = parseSchema(schema);
    expect(parsed).toEqual(templates);
  });
  it('should omit denied nodes', () => {
    const parsed = parseSchema(schema, undefined, ['add']);
    expect(parsed).toEqual(omit(templates, 'add'));
  });
  it('should include only allowed nodes', () => {
    const parsed = parseSchema(schema, ['add']);
    expect(parsed).toEqual(pick(templates, 'add'));
  });
});
