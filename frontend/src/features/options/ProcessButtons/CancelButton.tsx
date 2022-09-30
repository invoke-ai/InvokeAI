import React from 'react';
import { MdCancel } from 'react-icons/md';
import { cancelProcessing } from '../../../app/socketio/actions';
import { useAppDispatch, useAppSelector } from '../../../app/store';
import SDIconButton from '../../../common/components/SDIconButton';
import { systemSelector } from '../../../common/hooks/useCheckParameters';

export default function CancelButton() {
  const dispatch = useAppDispatch();
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
