import { createSelector } from '@reduxjs/toolkit';
import { AddToBatchDropData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';
import { FaLayerGroup } from 'react-icons/fa';
import { useDispatch } from 'react-redux';
import GenericBoard from './GenericBoard';

const selector = createSelector(stateSelector, (state) => {
  return {
    count: state.batch.ids.length,
  };
});

const BatchBoard = ({ isSelected }: { isSelected: boolean }) => {
  const dispatch = useDispatch();
  const { count } = useAppSelector(selector);

  const handleBatchBoardClick = useCallback(() => {
    dispatch(boardIdSelected('batch'));
  }, [dispatch]);

  const droppableData: AddToBatchDropData = {
    id: 'batch-board',
    actionType: 'ADD_TO_BATCH',
  };

  return (
    <GenericBoard
      droppableData={droppableData}
      onClick={handleBatchBoardClick}
      isSelected={isSelected}
      icon={FaLayerGroup}
      label="Batch"
      badgeCount={count}
    />
  );
};

export default BatchBoard;
