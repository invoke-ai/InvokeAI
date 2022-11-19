import UnifiedCanvasPanel from './UnifiedCanvasPanel';
import UnifiedCanvasDisplay from './UnifiedCanvasDisplay';
import InvokeWorkarea from 'features/tabs/components/InvokeWorkarea';
import { useAppDispatch } from 'app/store';
import { useEffect } from 'react';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';

export default function UnifiedCanvasWorkarea() {
  const dispatch = useAppDispatch();
  useEffect(() => {
    dispatch(setDoesCanvasNeedScaling(true));
  }, [dispatch]);
  return (
    <InvokeWorkarea
      optionsPanel={<UnifiedCanvasPanel />}
      styleClass="inpainting-workarea-overrides"
    >
      <UnifiedCanvasDisplay />
    </InvokeWorkarea>
  );
}
