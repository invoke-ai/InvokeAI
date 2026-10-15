/** Operable elements, minus the ones kept out of the tab order -- a select's hidden native control, for one. */
const FOCUSABLE = ['button', '[href]', 'input', 'select', 'textarea', '[tabindex]']
  .map((selector) => `${selector}:not([disabled]):not([tabindex="-1"]):not([aria-hidden="true"])`)
  .join(', ');

/** Moves focus to the first operable element inside `element`. */
export const focusFirstOperable = (element: HTMLElement | null): void => {
  element?.querySelector<HTMLElement>(FOCUSABLE)?.focus();
};

/** Restore focus after a control unmounts only while focus remains unclaimed; never steal a user's newer focus. */
export const focusIfUnclaimed = (element: HTMLElement | null): void => {
  if (document.activeElement !== null && document.activeElement !== document.body) {
    return;
  }

  focusFirstOperable(element);
};
