/** Invoke keeps full control corners while it sits inside its attached cluster. */

/** Doubled specificity outranks the attached Group's `!important` corner reset. */
export const INVOKE_BUTTON_CSS = { '&&': { borderRadius: 'control !important' } } as const;
