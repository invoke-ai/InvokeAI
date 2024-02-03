import type { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { isRejectedWithValue } from '@reduxjs/toolkit';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { z } from 'zod';

const zRejectedForbiddenAction = z.object({
  payload: z.object({
    status: z.literal(403),
    data: z.object({
      detail: z.string(),
    }),
  }),
  meta: z
    .object({
      arg: z
        .object({
          endpointName: z.string().optional(),
        })
        .optional(),
    })
    .optional(),
});

export const authToastMiddleware: Middleware = (api: MiddlewareAPI) => (next) => (action) => {
  if (isRejectedWithValue(action)) {
    try {
      const parsed = zRejectedForbiddenAction.parse(action);
      const endpointName = parsed.meta?.arg?.endpointName;
      if (endpointName === 'getImageDTO') {
        // do not show toast if problem is image access
        return;
      }

      const { dispatch } = api;
      const customMessage = parsed.payload.data.detail !== 'Forbidden' ? parsed.payload.data.detail : undefined;
      dispatch(
        addToast({
          id: `auth-error-toast-${endpointName}`,
          title: t('common.somethingWentWrong'),
          status: 'error',
          description: customMessage,
        })
      );
    } catch (error) {
      // no-op
    }
  }

  return next(action);
};
