import type { KeyboardEvent } from 'react';

const isEditingTarget = (target: EventTarget | null): boolean =>
  target instanceof HTMLElement &&
  (target.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName));

/**
 * `/` inside a settings surface jumps to its search field, unless a field is being edited or the field is out of
 * view (a single-pane page showing a section), where the key is left alone.
 */
export const focusSettingsSearchOnSlash = (event: KeyboardEvent<HTMLElement>) => {
  if (event.key !== '/' || event.ctrlKey || event.metaKey || event.altKey || isEditingTarget(event.target)) {
    return;
  }

  const search = event.currentTarget.querySelector<HTMLInputElement>('[data-settings-search]');

  if (search?.checkVisibility({ visibilityProperty: true })) {
    event.preventDefault();
    search.focus();
    search.select();
  }
};
