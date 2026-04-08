import { omit, pick } from 'es-toolkit/compat';
import { call_saved_workflow, schema, templates } from 'features/nodes/store/util/testUtils';
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
  it('should parse the call_saved_workflow node template', () => {
    const parsed = parseSchema(schema);
    expect(stripUndefinedDeep(parsed.call_saved_workflow)).toEqual(stripUndefinedDeep(call_saved_workflow));
    expect(parsed.call_saved_workflow.inputs.workflow_id.type.name).toBe('SavedWorkflowField');
    expect(parsed.call_saved_workflow.inputs.workflow_id.ui_type).toBe('SavedWorkflowField');
  });
});
