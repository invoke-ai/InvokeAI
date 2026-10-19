import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, createLogger, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { beforeEach, describe, expect, it } from 'vitest';

import { countProblems } from './useProblemCount';

beforeEach(() => {
  resetLogging();
  configureLogging({ ...DEFAULT_LOGGING_CONFIG, level: 'trace' });
});

describe('countProblems', () => {
  it('counts warnings and above for the project and unattributed application events only', () => {
    createLogger({ area: 'browser', namespace: 'app' }).error('uncaught');
    createLogger({ area: 'autosave', namespace: 'persistence', projectId: 'project-a' }).warn('slow');
    createLogger({ area: 'autosave', namespace: 'persistence', projectId: 'project-a' }).info('saved');
    createLogger({ area: 'history', namespace: 'queue', projectId: 'project-b' }).error('other project');

    expect(countProblems(getLogSnapshot().entries, 'project-a')).toBe(2);
    expect(countProblems(getLogSnapshot().entries, 'project-b')).toBe(2);
    expect(countProblems(getLogSnapshot().entries, 'project-c')).toBe(1);
  });
});
