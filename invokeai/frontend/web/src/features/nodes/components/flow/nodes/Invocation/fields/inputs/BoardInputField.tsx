import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import { fieldBoardValueChanged } from 'features/nodes/store/nodesSlice';
import {
  BoardInputFieldTemplate,
  BoardInputFieldValue,
  FieldComponentProps,
} from 'features/nodes/types/types';
import { memo, useCallback } from 'react';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

const BoardInputFieldComponent = (
  props: FieldComponentProps<BoardInputFieldValue, BoardInputFieldTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();

  const { data, hasBoards } = useListAllBoardsQuery(undefined, {
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
        data: boards,
        hasBoards: boards.length > 1,
      };
    },
  });

  const handleChange = useCallback(
    (v: string | null) => {
      dispatch(
        fieldBoardValueChanged({
          nodeId,
          fieldName: field.name,
          value: v && v !== 'none' ? { board_id: v } : undefined,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <IAIMantineSearchableSelect
      className="nowheel nodrag"
      value={field.value?.board_id ?? 'none'}
      data={data}
      onChange={handleChange}
      disabled={!hasBoards}
    />
  );
};

export default memo(BoardInputFieldComponent);
