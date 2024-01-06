import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { fieldBoardValueChanged } from 'features/nodes/store/nodesSlice';
import type {
  BoardFieldInputInstance,
  BoardFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

import type { FieldComponentProps } from './types';

const BoardFieldInputComponent = (
  props: FieldComponentProps<BoardFieldInputInstance, BoardFieldInputTemplate>
) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
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
      dispatch(
        fieldBoardValueChanged({
          nodeId,
          fieldName: field.name,
          value: v.value !== 'none' ? { board_id: v.value } : undefined,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const value = useMemo(
    () => options.find((o) => o.value === field.value?.board_id),
    [options, field.value]
  );

  const noOptionsMessage = useCallback(() => t('boards.noMatching'), [t]);

  return (
    <InvControl className="nowheel nodrag" isDisabled={!hasBoards}>
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

export default memo(BoardFieldInputComponent);
