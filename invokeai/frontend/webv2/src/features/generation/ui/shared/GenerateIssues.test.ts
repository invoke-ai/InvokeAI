import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { GENERATE_MODEL_SWITCH_SUBJECTS } from '@features/generation/core/baseGenerationPolicies';
import { collectGenerationIssueScenarios } from '@features/generation/core/generationIssues.testing';
import { describe, expect, it } from 'vitest';

import { ISSUE_FIELD_RENDERERS } from './GenerateIssues';

seedArchitectureCapabilities();

describe('ISSUE_FIELD_RENDERERS', () => {
  it('has a renderer for every field an issue points at; component slots render their own', () => {
    const componentSlots = new Set<string>(GENERATE_MODEL_SWITCH_SUBJECTS);
    const fields = collectGenerationIssueScenarios().flatMap(({ target }) => (target.field ? [target.field] : []));

    expect(fields.length).toBeGreaterThan(0);
    for (const field of fields) {
      expect(field in ISSUE_FIELD_RENDERERS || componentSlots.has(field), field).toBe(true);
    }
  });
});
