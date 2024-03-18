import type { UseToastOptions } from '@invoke-ai/ui-library';

export type MakeToastArg = string | UseToastOptions;

/**
 * Makes a toast from a string or a UseToastOptions object.
 * If a string is passed, the toast will have the status 'info' and will be closable with a duration of 2500ms.
 */
export const makeToast = (arg: MakeToastArg): UseToastOptions => {
  if (typeof arg === 'string') {
    return {
      title: arg,
      status: 'info',
      isClosable: true,
      duration: 2500,
    };
  }

  return { status: 'info', isClosable: true, duration: 2500, ...arg };
};
