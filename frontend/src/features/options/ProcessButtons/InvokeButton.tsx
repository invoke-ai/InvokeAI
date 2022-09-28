import React from 'react';
import { MdAddAPhoto } from 'react-icons/md';
import { useDispatch } from 'react-redux';
import { generateImage } from '../../../app/socketio/actions';
import SDIconButton from '../../../common/components/SDIconButton';
import useCheckParameters from '../../../common/hooks/useCheckParameters';

export default function InvokeButton() {
  const dispatch = useDispatch();
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
