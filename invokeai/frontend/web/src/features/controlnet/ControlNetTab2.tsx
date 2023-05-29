import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ControlNetTab from './ControlNetTab/ControlNetTab';
import { setControlNet2 } from './store/controlnetSlice';

export default function ControlNetTab2() {
  const controlnet2 = useAppSelector(
    (state: RootState) => state.controlnet.controlnet2
  );
  return (
    <ControlNetTab
      label="ControlNet 2"
      controlnet={controlnet2}
      setControlnet={setControlNet2}
    />
  );
}
