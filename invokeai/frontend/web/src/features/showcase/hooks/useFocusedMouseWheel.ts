import type { MutableRefObject } from 'react';
import { useEffect } from 'react';

export const useFocusedMouseWheel = (
  ref: MutableRefObject<HTMLDivElement | null>,
  trigger: (e: WheelEvent, ...args: unknown[]) => void
) => {
  useEffect(() => {
    const refElement = ref.current;
    if (refElement) {
      refElement.addEventListener('wheel', trigger);
      refElement.addEventListener('focus', () => {
        refElement.addEventListener('wheel', trigger);
      });
      refElement.addEventListener('blur', () => {
        refElement.removeEventListener('wheel', trigger);
      });
    }
    return () => {
      if (refElement) {
        refElement.removeEventListener('wheel', trigger);
        refElement.removeEventListener('focus', () => {
          refElement.removeEventListener('wheel', trigger);
        });
        refElement.removeEventListener('blur', () => {
          refElement.removeEventListener('wheel', trigger);
        });
      }
    };
  }, [trigger, ref]);
};
