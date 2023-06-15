import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { selectBoardsAll } from './boardSlice';

export const boardSelector = (state: RootState) => state.boards.entities;

export const searchBoardsSelector = createSelector(
  (state: RootState) => state,
  (state) => {
    const {
      boards: { searchText },
    } = state;

    if (!searchText) {
      // If no search text provided, return all entities
      return selectBoardsAll(state);
    }

    return selectBoardsAll(state).filter((i) =>
      i.board_name.toLowerCase().includes(searchText.toLowerCase())
    );
  }
);
