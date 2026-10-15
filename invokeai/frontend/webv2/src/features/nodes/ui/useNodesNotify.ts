import { toaster } from '@platform/ui';

export interface NodesNotify {
  error(title: string, message?: string): void;
  info(title: string, message?: string): void;
  success(title: string, message?: string): void;
  /** `sticky` keeps the toast until dismissed — for notices that demand action (e.g. manual dependency installs). */
  warning(title: string, message?: string, options?: { sticky?: boolean }): void;
}

const createNotice = (type: 'error' | 'info' | 'success') => (title: string, message?: string) => {
  toaster.create({ description: message, title, type });
};

const nodesNotify: NodesNotify = {
  error: createNotice('error'),
  info: createNotice('info'),
  success: createNotice('success'),
  warning: (title, message, options) => {
    toaster.create({
      description: message,
      duration: options?.sticky ? Number.POSITIVE_INFINITY : undefined,
      title,
      type: 'warning',
    });
  },
};

export const useNotify = (): NodesNotify => nodesNotify;
