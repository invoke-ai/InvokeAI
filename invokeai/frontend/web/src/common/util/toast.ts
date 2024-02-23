import { createStandaloneToast, theme, TOAST_OPTIONS } from '@invoke-ai/ui-library';

export const { toast } = createStandaloneToast({
  theme: theme,
  defaultOptions: TOAST_OPTIONS.defaultOptions,
});
