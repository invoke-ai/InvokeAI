import OutpaintingPanel from './OutpaintingPanel';
import OutpaintingDisplay from './OutpaintingDisplay';
import InvokeWorkarea from 'features/tabs/InvokeWorkarea';
import { useAppDispatch } from 'app/store';
import { useEffect } from 'react';
import { setDoesCanvasNeedScaling } from 'features/canvas/canvasSlice';

export default function OutpaintingWorkarea() {
  const dispatch = useAppDispatch();
  useEffect(() => {
    dispatch(setDoesCanvasNeedScaling(true));
  }, [dispatch]);
  return (
    <InvokeWorkarea
      optionsPanel={<OutpaintingPanel />}
      styleClass="inpainting-workarea-overrides"
    >
      <OutpaintingDisplay />
    </InvokeWorkarea>
  );
}
