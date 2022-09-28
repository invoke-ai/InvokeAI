import React from 'react';
import { MdCancel } from 'react-icons/md';
import { useDispatch } from 'react-redux';
import { cancelProcessing } from '../../../app/socketio/actions';
import { useAppSelector } from '../../../app/store';
import SDIconButton from '../../../common/components/SDIconButton';
import { systemSelector } from '../../../common/hooks/useCheckParameters';

export default function CancelButton() {
  const dispatch = useDispatch();
  const { isProcessing, isConnected } = useAppSelector(systemSelector);
  const handleClickCancel = () => dispatch(cancelProcessing());

  return (
    <SDIconButton
      icon={<MdCancel />}
      tooltip="Cancel"
      aria-label="Cancel"
      isDisabled={!isConnected || !isProcessing}
      onClick={handleClickCancel}
      className="cancel-btn"
    />
  );
}
