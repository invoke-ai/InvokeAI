export type WorkspaceClearFailure = 'projects' | 'settings';

const WORKSPACE_CLEAR_FAILURE_KEY = 'invokeai:v7:webv2:workspace-clear-failure';

export const rememberWorkspaceClearFailure = (message: string, storage: Pick<Storage, 'setItem'>): void => {
  try {
    storage.setItem(WORKSPACE_CLEAR_FAILURE_KEY, message);
  } catch {
    return;
  }
};

export const consumeWorkspaceClearFailure = (storage: Pick<Storage, 'getItem' | 'removeItem'>): string | null => {
  try {
    const message = storage.getItem(WORKSPACE_CLEAR_FAILURE_KEY);
    storage.removeItem(WORKSPACE_CLEAR_FAILURE_KEY);
    return message;
  } catch {
    return null;
  }
};

export const clearWorkspaceData = async (
  clearProjects: () => Promise<void>,
  clearSettings: () => Promise<void>
): Promise<WorkspaceClearFailure[]> => {
  const [projects, settings] = await Promise.allSettled([clearProjects(), clearSettings()]);
  return [
    ...(projects.status === 'rejected' ? (['projects'] as const) : []),
    ...(settings.status === 'rejected' ? (['settings'] as const) : []),
  ];
};
