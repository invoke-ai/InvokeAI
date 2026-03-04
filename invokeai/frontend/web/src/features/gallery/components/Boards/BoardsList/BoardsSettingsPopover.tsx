import type { ComboboxOption, IconButtonProps } from '@invoke-ai/ui-library';
import {
  Combobox,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Switch,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SingleValue } from 'chakra-react-select';
import {
  autoAssignBoardOnClickChanged,
  boardsListOrderByChanged,
  boardsListOrderDirChanged,
  selectGallerySlice,
  shouldShowArchivedBoardsChanged,
} from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixFill } from 'react-icons/pi';
import { assert } from 'tsafe';

const selectBoardsSettings = createSelector(selectGallerySlice, (gallery) => ({
  autoAssignBoardOnClick: gallery.autoAssignBoardOnClick,
  shouldShowArchivedBoards: gallery.shouldShowArchivedBoards,
  boardsListOrderBy: gallery.boardsListOrderBy,
  boardsListOrderDir: gallery.boardsListOrderDir,
}));

export const BoardsSettingsPopover = memo((iconButtonProps: Partial<IconButtonProps>) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { autoAssignBoardOnClick, shouldShowArchivedBoards, boardsListOrderBy, boardsListOrderDir } =
    useAppSelector(selectBoardsSettings);

  const orderByOptions = useMemo<ComboboxOption[]>(
    () => [
      { value: 'created_at', label: t('common.created') },
      { value: 'board_name', label: t('common.name', { defaultValue: 'Name' }) },
    ],
    [t]
  );

  const orderDirOptions = useMemo<ComboboxOption[]>(
    () => [
      { value: 'ASC', label: t('queue.sortOrderAscending') },
      { value: 'DESC', label: t('queue.sortOrderDescending') },
    ],
    [t]
  );

  const onChangeShowArchivedBoards = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(shouldShowArchivedBoardsChanged(e.target.checked)),
    [dispatch]
  );

  const onChangeAutoAssignBoardOnClick = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(autoAssignBoardOnClickChanged(e.target.checked)),
    [dispatch]
  );

  const onChangeOrderBy = useCallback(
    (v: SingleValue<ComboboxOption>) => {
      assert(v?.value === 'created_at' || v?.value === 'board_name');
      dispatch(boardsListOrderByChanged(v.value));
    },
    [dispatch]
  );

  const onChangeOrderDir = useCallback(
    (v: SingleValue<ComboboxOption>) => {
      assert(v?.value === 'ASC' || v?.value === 'DESC');
      dispatch(boardsListOrderDirChanged(v.value));
    },
    [dispatch]
  );

  const orderByValue = useMemo(
    () => orderByOptions.find((opt) => opt.value === boardsListOrderBy),
    [boardsListOrderBy, orderByOptions]
  );
  const orderDirValue = useMemo(
    () => orderDirOptions.find((opt) => opt.value === boardsListOrderDir),
    [boardsListOrderDir, orderDirOptions]
  );

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          size="sm"
          variant="ghost"
          icon={<PiGearSixFill />}
          aria-label={t('gallery.boardsSettings')}
          tooltip={t('gallery.boardsSettings')}
          {...iconButtonProps}
        />
      </PopoverTrigger>
      <Portal>
        <PopoverContent>
          <PopoverArrow />
          <PopoverBody>
            <Flex direction="column" gap={2}>
              <FormControl>
                <FormLabel flexGrow={1} m={0}>
                  {t('gallery.showArchivedBoards')}
                </FormLabel>
                <Switch size="sm" isChecked={shouldShowArchivedBoards} onChange={onChangeShowArchivedBoards} />
              </FormControl>
              <FormControl>
                <FormLabel flexGrow={1} m={0}>
                  {t('gallery.autoAssignBoardOnClick')}
                </FormLabel>
                <Switch size="sm" isChecked={autoAssignBoardOnClick} onChange={onChangeAutoAssignBoardOnClick} />
              </FormControl>
              <FormControl>
                <FormLabel flexGrow={1} m={0}>
                  {t('common.orderBy')}
                </FormLabel>
                <Combobox isSearchable={false} value={orderByValue} options={orderByOptions} onChange={onChangeOrderBy} />
              </FormControl>
              <FormControl>
                <FormLabel flexGrow={1} m={0}>
                  {t('common.direction')}
                </FormLabel>
                <Combobox isSearchable={false} value={orderDirValue} options={orderDirOptions} onChange={onChangeOrderDir} />
              </FormControl>
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

BoardsSettingsPopover.displayName = 'BoardsSettingsPopover';
