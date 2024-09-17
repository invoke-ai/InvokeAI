import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectAutoAddBoardId, selectAutoAssignBoardOnClick } from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

const BoardAutoAddSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);
  const { options, hasBoards } = useListAllBoardsQuery(
    {},
    {
      selectFromResult: ({ data }) => {
        const options: ComboboxOption[] = [
          {
            label: t('common.none'),
            value: 'none',
          },
        ].concat(
          (data ?? []).map(({ board_id, board_name }) => ({
            label: board_name,
            value: board_id,
          }))
        );
        return {
          options,
          hasBoards: options.length > 1,
        };
      },
    }
  );

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      dispatch(autoAddBoardIdChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === autoAddBoardId), [options, autoAddBoardId]);

  const noOptionsMessage = useCallback(() => t('boards.noMatching'), [t]);

  return (
    <FormControl isDisabled={!hasBoards || autoAssignBoardOnClick}>
      <FormLabel>{t('boards.autoAddBoard')}</FormLabel>
      <Combobox
        value={value}
        options={options}
        onChange={onChange}
        placeholder={t('boards.selectBoard')}
        noOptionsMessage={noOptionsMessage}
      />
    </FormControl>
  );
};
export default memo(BoardAutoAddSelect);
