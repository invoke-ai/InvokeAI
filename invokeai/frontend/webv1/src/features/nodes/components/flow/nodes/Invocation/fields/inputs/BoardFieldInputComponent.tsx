import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldBoardValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { BoardFieldInputInstance, BoardFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

import type { FieldComponentProps } from './types';

/**
 * The board field values in the UI do not map 1-to-1 to the values the graph expects.
 *
 * The graph value is either an object in the shape of `{board_id: string}` or undefined.
 *
 * But in the UI, we have the following options:
 * - auto: Use the "auto add" board. During graph building, we pull the auto add board ID from the state and use it.
 * - none: Do not assign a board. In the graph, this is represented as undefined.
 * - board_id: Assign the specified board. In the graph, this is represented as `{board_id: string}`.
 *
 * It's also possible that the UI value is undefined, which may be the case for some older workflows. In this case, we
 * map it to the "auto" option.
 *
 * So there is some translation that needs to happen in both directions - when the user selects a board in the UI, and
 * when we build the graph. The former is handled in this component, the latter in the `buildNodesGraph` function.
 */

const listAllBoardsQueryArg = { include_archived: true };

const getBoardValue = (val: string) => {
  if (val === 'auto' || val === 'none') {
    return val;
  }

  return {
    board_id: val,
  };
};

const BoardFieldInputComponent = (props: FieldComponentProps<BoardFieldInputInstance, BoardFieldInputTemplate>) => {
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const listAllBoardsQuery = useListAllBoardsQuery(listAllBoardsQueryArg);

  const autoOption = useMemo<ComboboxOption>(() => {
    return {
      label: t('common.auto'),
      value: 'auto',
    };
  }, [t]);

  const noneOption = useMemo<ComboboxOption>(() => {
    return {
      label: `${t('common.none')} (${t('boards.uncategorized')})`,
      value: 'none',
    };
  }, [t]);

  const options = useMemo<ComboboxOption[]>(() => {
    const _options: ComboboxOption[] = [autoOption, noneOption];
    if (listAllBoardsQuery.data) {
      for (const board of listAllBoardsQuery.data) {
        _options.push({
          label: board.board_name,
          value: board.board_id,
        });
      }
    }
    return _options;
  }, [autoOption, listAllBoardsQuery.data, noneOption]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        // This should never happen
        return;
      }

      const value = getBoardValue(v.value);

      dispatch(
        fieldBoardValueChanged({
          nodeId,
          fieldName: field.name,
          value,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const value = useMemo(() => {
    const _value = field.value;
    if (!_value || _value === 'auto') {
      return autoOption;
    }
    if (_value === 'none') {
      return noneOption;
    }
    const boardOption = options.find((o) => o.value === _value.board_id);
    return boardOption ?? autoOption;
  }, [field.value, options, autoOption, noneOption]);

  const noOptionsMessage = useCallback(() => t('boards.noMatching'), [t]);

  return (
    <Combobox
      className={`${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`}
      value={value}
      options={options}
      onChange={onChange}
      placeholder={t('boards.selectBoard')}
      noOptionsMessage={noOptionsMessage}
    />
  );
};

export default memo(BoardFieldInputComponent);
