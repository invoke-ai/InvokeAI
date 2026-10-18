import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { requestNewProject, takeNewProjectRequest, type NewProjectRequest } from './newProjectRequest';

const request = (name: string | null = 'Portraits'): NewProjectRequest => ({
  generateSettings: { height: 840, width: 1256 },
  intent: 'canvas',
  libraryWorkflow: null,
  name,
  startImage: null,
});

beforeEach(() => {
  accountLifecycle.activate('user-a', ':user:user-a');
});

afterEach(() => {
  accountLifecycle.invalidate();
});

describe('newProjectRequest', () => {
  it('hands the staged request out exactly once', () => {
    const staged = request();

    requestNewProject(staged);

    expect(takeNewProjectRequest()).toBe(staged);
    expect(takeNewProjectRequest()).toBeNull();
  });

  it('keeps only the latest request', () => {
    requestNewProject(request('First'));
    requestNewProject(request('Second'));

    expect(takeNewProjectRequest()?.name).toBe('Second');
  });

  it('drops a staged request when the account changes', () => {
    requestNewProject(request());

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b', ':user:user-b');

    expect(takeNewProjectRequest()).toBeNull();
  });
});
