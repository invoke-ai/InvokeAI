import { t } from 'i18next';

type ApiErrorLike = {
  data?: {
    detail?: string;
  };
};

export const getApiErrorDetail = (error: unknown): string => {
  if (typeof error === 'object' && error !== null && 'data' in error) {
    const { data } = error as ApiErrorLike;
    if (typeof data?.detail === 'string') {
      return data.detail;
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return t('common.unknown');
};
