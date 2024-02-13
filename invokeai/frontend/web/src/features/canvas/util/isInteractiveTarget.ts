import { isInputElement } from 'common/util/isInputElement';

export const isInteractiveTarget = (target: EventTarget | null) => {
  if (target instanceof HTMLElement) {
    return (
      target.tabIndex > -1 ||
      isInputElement(target) ||
      ['dialog', 'alertdialog'].includes(target.getAttribute('role') ?? '')
    );
  }

  return false;
};
