import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import SDSwitch from '../../../../common/components/SDSwitch';
import { setShouldFitToWidthHeight } from '../../optionsSlice';

export default function ImageFit() {
  const dispatch = useAppDispatch();

  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.options.shouldFitToWidthHeight
  );

  const handleChangeFit = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldFitToWidthHeight(e.target.checked));

  return (
    <SDSwitch
      label="Fit initial image to output size"
      isChecked={shouldFitToWidthHeight}
      onChange={handleChangeFit}
    />
  );
}
