import { apiFetchJson } from '@platform/transport/http';

import type { MainModelConfig } from './types';

export const listMainModels = async (): Promise<MainModelConfig[]> => {
  const body = await apiFetchJson<{ models?: MainModelConfig[] }>('/api/v2/models/?model_type=main');

  return body.models ?? [];
};
