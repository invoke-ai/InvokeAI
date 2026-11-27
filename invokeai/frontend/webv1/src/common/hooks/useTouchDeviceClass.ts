import { useEffect } from 'react';

const TOUCH_DEVICE_CLASS = 'invokeai-touch-device';

export const useTouchDeviceClass = () => {
  useEffect(() => {
    const onPointerInput = (e: PointerEvent) => {
      if (e.pointerType === 'touch') {
        document.documentElement.classList.add(TOUCH_DEVICE_CLASS);
      } else if (e.pointerType === 'mouse') {
        document.documentElement.classList.remove(TOUCH_DEVICE_CLASS);
      }
    };

    window.addEventListener('pointerdown', onPointerInput, { passive: true });
    window.addEventListener('pointermove', onPointerInput, { passive: true });

    return () => {
      window.removeEventListener('pointerdown', onPointerInput);
      window.removeEventListener('pointermove', onPointerInput);
    };
  }, []);
};
