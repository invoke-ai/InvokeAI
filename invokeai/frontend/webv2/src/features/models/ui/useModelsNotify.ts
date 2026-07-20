import { toaster } from '@platform/ui';

export interface ModelsNotify {
  error(title: string, message?: string): void;
  info(title: string, message?: string): void;
  success(title: string, message?: string): void;
}

const createNotice = (type: 'error' | 'info' | 'success') => (title: string, message?: string) => {
  toaster.create({ description: message, title, type });
};

const modelsNotify: ModelsNotify = {
  error: createNotice('error'),
  info: createNotice('info'),
  success: createNotice('success'),
};

export const useNotify = (): ModelsNotify => modelsNotify;
