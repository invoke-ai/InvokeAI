import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useHotkeys } from 'react-hotkeys-hook';

export const useStagingAreaKeyboardNav = () => {
  const ctx = useCanvasSessionContext();

  useHotkeys('left', ctx.selectPrev, { preventDefault: true });
  useHotkeys('right', ctx.selectNext, { preventDefault: true });
  useHotkeys('meta+left', ctx.selectFirst, { preventDefault: true });
  useHotkeys('meta+right', ctx.selectLast, { preventDefault: true });
};
