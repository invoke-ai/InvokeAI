import { describe, expect, it } from 'vitest';

import {
  EXPORT_PADDING,
  EXPORT_SCALE,
  getWorkflowImageDimensions,
  sanitizeWorkflowImageFilename,
} from './workflowImageExport';

describe('workflow image export', () => {
  it('uses padded logical bounds and export scale for output dimensions', () => {
    expect(getWorkflowImageDimensions({ x: -100, y: 50, width: 1600, height: 900 })).toEqual({
      width: 1600 + EXPORT_PADDING * 2,
      height: 900 + EXPORT_PADDING * 2,
      canvasWidth: (1600 + EXPORT_PADDING * 2) * EXPORT_SCALE,
      canvasHeight: (900 + EXPORT_PADDING * 2) * EXPORT_SCALE,
    });
  });

  it('keeps ordinary workflow names unchanged', () => {
    expect(sanitizeWorkflowImageFilename('Hi-Res Two Stage')).toBe('Hi-Res Two Stage');
  });

  it('replaces filesystem-invalid characters and falls back for blank names', () => {
    expect(sanitizeWorkflowImageFilename('Workflow: 01 / test?')).toBe('Workflow- 01 - test-');
    expect(sanitizeWorkflowImageFilename('   ...   ')).toBe('My Workflow');
  });
});
