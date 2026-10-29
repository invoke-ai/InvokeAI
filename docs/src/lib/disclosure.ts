type DisclosureOptions = {
  /** Runs after the popup opens, e.g. to move focus into it. */
  onOpen?: () => void;
};

/**
 * Wires a button that shows and hides a popup inside `root`: the button toggles it and keeps `aria-expanded` in sync;
 * Escape closes it and returns focus to the button; a click outside `root` or focus leaving `root` closes it.
 */
export const bindDisclosure = (
  root: HTMLElement,
  button: HTMLButtonElement,
  popup: HTMLElement,
  { onOpen }: DisclosureOptions = {}
) => {
  const isOpen = () => !popup.hidden;
  const setOpen = (open: boolean) => {
    popup.hidden = !open;
    button.setAttribute('aria-expanded', String(open));
    if (open) onOpen?.();
  };
  const close = ({ restoreFocus = false } = {}) => {
    if (!isOpen()) return;
    setOpen(false);
    if (restoreFocus) button.focus();
  };

  button.addEventListener('click', () => (isOpen() ? close() : setOpen(true)));
  root.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !isOpen()) return;
    // Cancelling the key press keeps it from also closing an enclosing popover, such as the mobile sidebar.
    event.preventDefault();
    event.stopPropagation();
    close({ restoreFocus: true });
  });
  document.addEventListener('click', (event) => {
    if (!root.contains(event.target as Node)) close();
  });
  root.addEventListener('focusout', (event) => {
    if (event.relatedTarget instanceof Node && !root.contains(event.relatedTarget)) close();
  });

  return { close, isOpen };
};
