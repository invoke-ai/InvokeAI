import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Box,
  Flex,
  Spinner,
  Text,
} from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';

import { memo, useContext, useRef, useState } from 'react';
import { AddImageToBoardContext } from '../../../../app/contexts/AddImageToBoardContext';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

const UpdateImageBoardModal = () => {
  // const boards = useSelector(selectBoardsAll);
  const { data: boards, isFetching } = useListAllBoardsQuery();
  const { isOpen, onClose, handleAddToBoard, image } = useContext(
    AddImageToBoardContext
  );
  const [selectedBoard, setSelectedBoard] = useState<string | null>();

  const cancelRef = useRef<HTMLButtonElement>(null);

  const currentBoard = boards?.find(
    (board) => board.board_id === image?.board_id
  );

  return (
    <AlertDialog
      isOpen={isOpen}
      leastDestructiveRef={cancelRef}
      onClose={onClose}
      isCentered
    >
      <AlertDialogOverlay>
        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {currentBoard ? 'Move Image to Board' : 'Add Image to Board'}
          </AlertDialogHeader>

          <AlertDialogBody>
            <Box>
              <Flex direction="column" gap={3}>
                {currentBoard && (
                  <Text>
                    Moving this image from{' '}
                    <strong>{currentBoard.board_name}</strong> to
                  </Text>
                )}
                {isFetching ? (
                  <Spinner />
                ) : (
                  <IAIMantineSelect
                    placeholder="Select Board"
                    onChange={(v) => setSelectedBoard(v)}
                    value={selectedBoard}
                    data={(boards ?? []).map((board) => ({
                      label: board.board_name,
                      value: board.board_id,
                    }))}
                  />
                )}
              </Flex>
            </Box>
          </AlertDialogBody>
          <AlertDialogFooter>
            <IAIButton onClick={onClose}>Cancel</IAIButton>
            <IAIButton
              isDisabled={!selectedBoard}
              colorScheme="accent"
              onClick={() => {
                if (selectedBoard) {
                  handleAddToBoard(selectedBoard);
                }
              }}
              ml={3}
            >
              {currentBoard ? 'Move' : 'Add'}
            </IAIButton>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(UpdateImageBoardModal);
