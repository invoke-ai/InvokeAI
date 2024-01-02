import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { autoAddBoardIdChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

const selector = createMemoizedSelector([stateSelector], ({ gallery }) => {
  const { autoAddBoardId, autoAssignBoardOnClick } = gallery;

  return {
    autoAddBoardId,
    autoAssignBoardOnClick,
  };
});

const BoardAutoAddSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { autoAddBoardId, autoAssignBoardOnClick } = useAppSelector(selector);
  const { options, hasBoards } = useListAllBoardsQuery(undefined, {
    selectFromResult: ({ data }) => {
      const options: InvSelectOption[] = [
        {
          label: 'None',
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
  });

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      dispatch(autoAddBoardIdChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => options.find((o) => o.value === autoAddBoardId),
    [options, autoAddBoardId]
  );

  const noOptionsMessage = useCallback(() => t('boards.noMatching'), [t]);

  return (
    <InvControl
      label={t('boards.autoAddBoard')}
      isDisabled={!hasBoards || autoAssignBoardOnClick}
    >
      <InvSelect
        value={value}
        options={options}
        onChange={onChange}
        placeholder={t('boards.selectBoard')}
        noOptionsMessage={noOptionsMessage}
      />
    </InvControl>
  );
};
export default memo(BoardAutoAddSelect);
