import { useEffect, useState } from 'react';

/** Tracks whether a keyboard modifier is currently held (e.g. Control for grid snapping). */
export const useModifierHeld = (key: 'Alt' | 'Control' | 'Meta' | 'Shift'): boolean => {
  const [isHeld, setIsHeld] = useState(false);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === key) {
        setIsHeld(true);
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === key) {
        setIsHeld(false);
      }
    };
    // Releasing the key outside the window would otherwise leave it stuck on.
    const onBlur = () => setIsHeld(false);

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, [key]);

  return isHeld;
};
