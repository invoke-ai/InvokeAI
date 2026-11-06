import { isImeComposing } from './imeComposition';

/** Popups whose typeahead takes printable keys. */
const TYPEAHEAD_SELECTOR = '[role="menu"], [role="menubar"], [role="listbox"]';

const isTypingTarget = (target: EventTarget | null): boolean =>
  target instanceof HTMLElement &&
  (target.isContentEditable || target.closest(`input, select, textarea, ${TYPEAHEAD_SELECTOR}`) !== null);

/**
 * A bare `/` jumps to the search field matching `selector` inside `container`: focuses and selects it and consumes
 * the key, so the slash is not typed into it. The key is left alone while something else takes it (a field being
 * typed in, a menu's or listbox's typeahead, an IME, a handler that already prevented it) and while the field is out
 * of view, so a hidden surface or a single-pane view showing its detail does not answer. Returns whether it did.
 */
export const focusSearchOnSlash = (event: KeyboardEvent, container: ParentNode, selector: string): boolean => {
  if (
    event.key !== '/' ||
    event.ctrlKey ||
    event.metaKey ||
    event.altKey ||
    event.defaultPrevented ||
    isImeComposing(event) ||
    isTypingTarget(event.target)
  ) {
    return false;
  }

  const field = container.querySelector<HTMLInputElement>(selector);

  if (!field?.checkVisibility({ visibilityProperty: true })) {
    return false;
  }

  event.preventDefault();
  field.focus();
  field.select();

  return true;
};
