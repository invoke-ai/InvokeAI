// Each registration holds its own token, so a release runs once however often it is called and never releases a
// registration that replaced it (StrictMode's mount, unmount, mount).
const presentModals = new Set<symbol>();
const listeners = new Set<() => void>();

export const subscribeModalPresence = (listener: () => void): (() => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

// A dialog consumes the key that closes it (Escape) in a document capture listener, and releases its presence before
// the same press reaches bubble-phase listeners. Key presses that began under a modal are remembered at window capture,
// which runs before any document listener, so the press still counts as aimed at the modal.
const keyDownsUnderModal = new WeakSet<Event>();
const rememberKeyDown = (event: Event) => {
  keyDownsUnderModal.add(event);
};

/** Marks an interactive modal as present until the returned release runs. */
export const registerModalPresence = (): (() => void) => {
  const token = Symbol('modal');

  if (presentModals.size === 0 && typeof window !== 'undefined') {
    window.addEventListener('keydown', rememberKeyDown, true);
  }
  presentModals.add(token);
  listeners.forEach((listener) => listener());

  return () => {
    if (presentModals.delete(token)) {
      if (presentModals.size === 0 && typeof window !== 'undefined') {
        window.removeEventListener('keydown', rememberKeyDown, true);
      }
      listeners.forEach((listener) => listener());
    }
  };
};

/**
 * Whether an interactive modal is present; read at the moment of use rather than mirrored into render state. Pass the
 * key event being handled to also count a modal that was present when that press began.
 */
export const isModalPresent = (keyDown?: Event): boolean =>
  presentModals.size > 0 || (keyDown !== undefined && keyDownsUnderModal.has(keyDown));
