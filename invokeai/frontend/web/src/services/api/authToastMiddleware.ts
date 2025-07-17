import type { Middleware } from '@reduxjs/toolkit';
import { isRejectedWithValue } from '@reduxjs/toolkit';
import { $toastMap } from 'app/store/nanostores/toastMap';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { z } from 'zod/v4';

const trialUsageErrorSubstring = 'usage allotment for the free trial';
const trialUsageErrorCode = 'USAGE_LIMIT_TRIAL';

const orgUsageErrorSubstring = 'organization has reached its predefined usage allotment';
const orgUsageErrorCode = 'USAGE_LIMIT_ORG';

const indieUsageErrorSubstring = 'usage allotment';
const indieUsageErrorCode = 'USAGE_LIMIT_INDIE';

//TODO make this dynamic with returned error codes instead of substring check
const getErrorCode = (errorString?: string) => {
  if (!errorString) {
    return undefined;
  }
  if (errorString.includes(trialUsageErrorSubstring)) {
    return trialUsageErrorCode;
  }
  if (errorString.includes(orgUsageErrorSubstring)) {
    return orgUsageErrorCode;
  }
  if (errorString.includes(indieUsageErrorSubstring)) {
    return indieUsageErrorCode;
  }
};

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

export const authToastMiddleware: Middleware = () => (next) => (action) => {
  if (isRejectedWithValue(action)) {
    try {
      const parsed = zRejectedForbiddenAction.parse(action);
      const endpointName = parsed.meta?.arg?.endpointName;
      if (endpointName === 'getImageDTO') {
        // do not show toast if problem is image access
        return next(action);
      }
      const toastMap = $toastMap.get();
      const customMessage = parsed.payload.data.detail !== 'Forbidden' ? parsed.payload.data.detail : undefined;
      const errorCode = getErrorCode(customMessage);
      const customToastConfig = errorCode ? toastMap?.[errorCode] : undefined;

      if (customToastConfig) {
        toast(customToastConfig);
      } else {
        toast({
          id: `auth-error-toast-${endpointName}`,
          title: t('toast.somethingWentWrong'),
          status: 'error',
          description: customMessage,
        });
      }
    } catch {
      // no-op
    }
  }

  return next(action);
};
