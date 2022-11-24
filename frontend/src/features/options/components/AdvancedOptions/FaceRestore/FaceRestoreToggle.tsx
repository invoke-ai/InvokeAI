import { ChangeEvent } from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldRunFacetool } from 'features/options/store/optionsSlice';

export default function FaceRestoreToggle() {
  const isGFPGANAvailable = useAppSelector(
    (state: RootState) => state.system.isGFPGANAvailable
  );

  const shouldRunFacetool = useAppSelector(
    (state: RootState) => state.options.shouldRunFacetool
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
