import InpaintingPanel from './InpaintingPanel';
import InpaintingDisplay from './InpaintingDisplay';
import InvokeWorkarea from 'features/tabs/InvokeWorkarea';
import { useAppDispatch } from 'app/store';
import { useEffect } from 'react';
import { setDoesCanvasNeedScaling } from 'features/canvas/canvasSlice';

export default function InpaintingWorkarea() {
  const dispatch = useAppDispatch();
  useEffect(() => {
    dispatch(setDoesCanvasNeedScaling(true));
  }, [dispatch]);
  return (
    <InvokeWorkarea
      optionsPanel={<InpaintingPanel />}
      styleClass="inpainting-workarea-overrides"
    >
      <InpaintingDisplay />
    </InvokeWorkarea>
  );
}
