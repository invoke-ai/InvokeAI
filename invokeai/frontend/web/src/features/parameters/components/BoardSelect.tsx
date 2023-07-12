import { Flex, Text, Box } from '@chakra-ui/react';
import IAIMantineSelect from '../../../common/components/IAIMantineSelect';
import { useCallback } from 'react';
import { useAppDispatch, useAppSelector } from '../../../app/store/storeHooks';
import { useListAllBoardsQuery } from '../../../services/api/endpoints/boards';
import { setBoardIdToAddTo } from '../../system/store/systemSlice';
import { RootState } from '../../../app/store/store';

const BoardSelect = () => {
  const { data: boards, isFetching } = useListAllBoardsQuery();
  const dispatch = useAppDispatch();

  const boardIdToAddTo = useAppSelector(
    (state: RootState) => state.system.boardIdToAddTo
  );

  const handleChange = useCallback(
    (v: string | '') => {
      if (v === '') {
        dispatch(setBoardIdToAddTo(undefined));
      } else {
        dispatch(setBoardIdToAddTo(v));
      }
    },
    [dispatch]
  );

  return (
    <Flex gap={2} alignItems="center">
      <Box>
        <Text
          sx={{
            color: 'base.400',
            fontSize: 'sm',
          }}
        >
          Add to board:
        </Text>
      </Box>
      <IAIMantineSelect
        style={{ flexGrow: 1 }}
        clearable={true}
        placeholder="None"
        onChange={handleChange}
        value={boardIdToAddTo}
        data={(boards ?? []).map((board) => ({
          label: board.board_name,
          value: board.board_id,
        }))}
      />
    </Flex>
  );
};

export default BoardSelect;
