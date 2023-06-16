import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Box,
  Divider,
  Flex,
  Select,
  Text,
} from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';

import { memo, useContext, useRef, useState } from 'react';
import { AddImageToBoardContext } from '../../../../app/contexts/AddImageToBoardContext';
import { useSelector } from 'react-redux';
import { selectBoardsAll } from '../../store/boardSlice';
import IAISelect from '../../../../common/components/IAISelect';
import IAIMantineSelect from 'common/components/IAIMantineSelect';

const UpdateImageBoardModal = () => {
  const boards = useSelector(selectBoardsAll);
  const [selectedBoard, setSelectedBoard] = useState<string | null>(null);

  const { isOpen, onClose, handleAddToBoard, image } = useContext(
    AddImageToBoardContext
  );

  const cancelRef = useRef<HTMLButtonElement>(null);

  const currentBoard = boards.filter(
    (board) => board.board_id === image?.board_id
  )[0];

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
            Move Image to Board
          </AlertDialogHeader>

          <AlertDialogBody>
            <Box>
              <Flex direction="column" gap={3}>
                <Text>
                  Moving this image to a board will remove it from its existing
                  board.
                </Text>
                <IAIMantineSelect
                  placeholder="Select Board"
                  onChange={(v) => setSelectedBoard(v)}
                  value={selectedBoard}
                  data={boards.map((board) => ({
                    label: board.board_name,
                    value: board.board_id,
                  }))}
                />
              </Flex>
            </Box>
          </AlertDialogBody>
          <AlertDialogFooter>
            <IAIButton onClick={onClose}>Cancel</IAIButton>
            <IAIButton
              isDisabled={!selectedBoard}
              colorScheme="accent"
              onClick={() => {
                if (selectedBoard) handleAddToBoard(selectedBoard);
              }}
              ml={3}
            >
              Add to Board
            </IAIButton>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(UpdateImageBoardModal);
