import type { RefusedWorkbenchProject } from '@workbench/projectContracts';

import { MIN_SUPPORTED_CANVAS_SCHEMA_VERSION } from '@workbench/canvasSchemaVersion';
import { describe, expect, it } from 'vitest';

import { toInvkRefusalError } from './format';

const refused = (source: 'canvas' | 'project-document', version: number) =>
  ({
    projectId: 'p',
    projectName: 'P',
    raw: {},
    refusal: { raw: {}, scope: source, status: 'unsupported-version', version },
    source,
  }) as unknown as RefusedWorkbenchProject;

describe('toInvkRefusalError', () => {
  it('names which schema refused the document, and an old canvas as retired rather than newer', () => {
    expect(toInvkRefusalError(refused('project-document', 5))).toMatchObject({
      message: 'Project requires document schema 5.',
      reason: 'unsupported-version',
    });
    expect(toInvkRefusalError(refused('canvas', 99))).toMatchObject({
      message: 'Project requires canvas schema 99.',
      reason: 'unsupported-version',
    });
    expect(toInvkRefusalError(refused('canvas', MIN_SUPPORTED_CANVAS_SCHEMA_VERSION - 1))).toMatchObject({
      reason: 'legacy-canvas-project',
    });
  });
});
