import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldRunFacetool } from 'features/parameters/store/postprocessingSlice';
import { ChangeEvent } from 'react';

export default function FaceRestoreToggle() {
  const isGFPGANAvailable = useAppSelector(
    (state: RootState) => state.system.isGFPGANAvailable
  );

  const shouldRunFacetool = useAppSelector(
    (state: RootState) => state.postprocessing.shouldRunFacetool
  );

  const dispatch = useAppDispatch();

  const handleChangeShouldRunFacetool = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRunFacetool(e.target.checked));

  return (
    <IAISwitch
      isDisabled={!isGFPGANAvailable}
      isChecked={shouldRunFacetool}
      onChange={handleChangeShouldRunFacetool}
    />
  );
}
