import { SelectItem } from '@mantine/core';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

const selector = createSelector(
  [stateSelector],
  ({ gallery }) => {
    const { autoAddBoardId, autoAssignBoardOnClick } = gallery;

    return {
      autoAddBoardId,
      autoAssignBoardOnClick,
    };
  },
  defaultSelectorOptions
);

const BoardAutoAddSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { autoAddBoardId, autoAssignBoardOnClick } = useAppSelector(selector);
  const inputRef = useRef<HTMLInputElement>(null);
  const { boards, hasBoards } = useListAllBoardsQuery(undefined, {
    selectFromResult: ({ data }) => {
      const boards: SelectItem[] = [
        {
          label: 'None',
          value: 'none',
        },
      ];
      data?.forEach(({ board_id, board_name }) => {
        boards.push({
          label: board_name,
          value: board_id,
        });
      });
      return {
        boards,
        hasBoards: boards.length > 1,
      };
    },
  });

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      dispatch(autoAddBoardIdChanged(v));
    },
    [dispatch]
  );

  const filterFunc = useCallback(
    (value: string, item: SelectItem) =>
      item.label?.toLowerCase().includes(value.toLowerCase().trim()) ||
      item.value.toLowerCase().includes(value.toLowerCase().trim()),
    []
  );

  return (
    <IAIMantineSearchableSelect
      label={t('boards.autoAddBoard')}
      inputRef={inputRef}
      autoFocus
      placeholder={t('boards.selectBoard')}
      value={autoAddBoardId}
      data={boards}
      nothingFound={t('boards.noMatching')}
      itemComponent={IAIMantineSelectItemWithTooltip}
      disabled={!hasBoards || autoAssignBoardOnClick}
      filter={filterFunc}
      onChange={handleChange}
    />
  );
};

export default memo(BoardAutoAddSelect);
