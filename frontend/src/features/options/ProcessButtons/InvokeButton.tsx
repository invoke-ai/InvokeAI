import React from 'react';
import { MdAddAPhoto } from 'react-icons/md';
import { generateImage } from '../../../app/socketio/actions';
import { useAppDispatch } from '../../../app/store';
import SDIconButton from '../../../common/components/SDIconButton';
import useCheckParameters from '../../../common/hooks/useCheckParameters';

export default function InvokeButton() {
  const dispatch = useAppDispatch();
  const isReady = useCheckParameters();

  const handleClickGenerate = () => {
    dispatch(generateImage());
  };

  return (
    <SDIconButton
      icon={<MdAddAPhoto />}
      tooltip="Invoke"
      aria-label="Invoke"
      type="submit"
      isDisabled={!isReady}
      onClick={handleClickGenerate}
      className="invoke-btn"
    />
  );
}
