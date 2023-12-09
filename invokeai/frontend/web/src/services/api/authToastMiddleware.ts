import type { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { isRejectedWithValue } from '@reduxjs/toolkit';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { z } from 'zod';

const zRejectedForbiddenAction = z.object({
  action: z.object({
    payload: z.object({
      status: z.literal(403),
      data: z.object({
        detail: z.string(),
      }),
    }),
  }),
});

export const authToastMiddleware: Middleware =
  (api: MiddlewareAPI) => (next) => (action) => {
    if (isRejectedWithValue(action)) {
      try {
        const parsed = zRejectedForbiddenAction.parse(action);
        const { dispatch } = api;
        const customMessage =
          parsed.action.payload.data.detail !== 'Forbidden'
            ? parsed.action.payload.data.detail
            : undefined;
        dispatch(
          addToast({
            title: t('common.somethingWentWrong'),
            status: 'error',
            description: customMessage,
          })
        );
      } catch {
        // no-op
      }
    }

    return next(action);
  };
