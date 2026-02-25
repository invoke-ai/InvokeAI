import {
  type BoxProps,
  Button,
  Flex,
  Image,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Spacer,
  type SystemStyleObject,
  Text,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { getRegex, Picker, usePickerContext } from 'common/components/Picker/Picker';
import { useDisclosure } from 'common/hooks/useBoolean';
import {
  selectAutoAddBoardId,
  selectAutoAssignBoardOnClick,
  selectBoardSearchText,
  selectListBoardsQueryArgs,
  selectSelectedBoardId,
} from 'features/gallery/store/gallerySelectors';
import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { BoardDTO } from 'services/api/types';

const isMatch = (board: BoardDTO, searchTerm: string) => {
  const regex = getRegex(searchTerm);
  const testString = `${board.board_name}`.toLowerCase();

  if (testString.includes(searchTerm) || regex.test(testString)) {
    return true;
  }

  return false;
};

export const BoardsDropdown = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const popover = useDisclosure(false);
  const pickerRef = useRef(null);

  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const autoAssignBoardOnClick = useAppSelector(selectAutoAssignBoardOnClick);

  const boardSearchText = useAppSelector(selectBoardSearchText);
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const queryArgs = useAppSelector(selectListBoardsQueryArgs);

  const { data: boards } = useListAllBoardsQuery(queryArgs);

  const selectedBoardData = boards?.find((board) => board.board_id === selectedBoardId);

  const onClose = useCallback(() => {
    popover.close();
  }, [popover]);

  const getOptionId = useCallback((option: BoardDTO) => option.board_id, []);

  const onOptionSelect = useCallback(
    (option: BoardDTO) => {
      if (selectedBoardId !== option.board_id) {
        dispatch(boardIdSelected({ boardId: option.board_id }));
      }
      if (autoAssignBoardOnClick && autoAddBoardId !== option.board_id) {
        dispatch(autoAddBoardIdChanged(option.board_id));
      }
    },
    [selectedBoardId, autoAssignBoardOnClick, autoAddBoardId, dispatch]
  );

  return (
    <Popover
      isOpen={popover.isOpen}
      onOpen={popover.open}
      onClose={onClose}
      // initialFocusRef={switcherRef.current?.inputRef}
    >
      <PopoverTrigger>
        <Button size="sm" variant="outline" isDisabled={!boards || boards.length === 0} width="100%" maxWidth="200px">
          {selectedBoardData?.board_name ?? t('boards.uncategorized')}
          <Spacer />
          <PiCaretDownBold />
        </Button>
      </PopoverTrigger>
      <Portal appendToParentPortal={false}>
        <PopoverContent p={0} w={400} h={400}>
          <PopoverArrow />
          <PopoverBody p={0} w="full" h="full" borderWidth={1} borderColor="base.700" borderRadius="base">
            <Picker
              pickerId="boards-picker"
              handleRef={pickerRef}
              optionsOrGroups={boards ?? EMPTY_ARRAY}
              getOptionId={getOptionId}
              isMatch={isMatch}
              OptionComponent={BoardsSwitcherOptionComponent}
              onSelect={onOptionSelect}
              selectedOption={selectedBoardData}
              NextToSearchBar={undefined}
              searchable
              searchTerm={boardSearchText}
            />
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

BoardsDropdown.displayName = 'BoardsDropdown';

const optionSx: SystemStyleObject = {
  p: 1,
  gap: 2,
  alignItems: 'center',
  cursor: 'pointer',
  borderRadius: 'base',
  '&[data-selected="true"]': {
    bg: 'invokeBlue.300',
    color: 'base.900',
    '.extra-info': {
      color: 'base.700',
    },
    '.picker-option': {
      fontWeight: 'bold',
      '&[data-is-compact="true"]': {
        fontWeight: 'semibold',
      },
    },
    '&[data-active="true"]': {
      bg: 'invokeBlue.250',
    },
  },
  '&[data-active="true"]': {
    bg: 'base.750',
  },
  '&[data-disabled="true"]': {
    cursor: 'not-allowed',
    opacity: 0.5,
  },
  '&[data-is-compact="true"]': {
    px: 1,
    py: 0.5,

    '& img': {
      w: 8,
      h: 8,
    },
  },
  scrollMarginTop: '24px', // magic number, this is the height of the header
};

const BoardsSwitcherOptionComponent = memo(
  ({
    option,
    ...rest
  }: {
    option: BoardDTO;
  } & BoxProps) => {
    const { board_name, cover_image_name } = option;
    const { currentData: coverImage } = useGetImageDTOQuery(cover_image_name ?? skipToken);
    const { isCompactView } = usePickerContext();

    return (
      <Flex {...rest} sx={optionSx} data-is-compact={isCompactView}>
        <Image
          src={coverImage?.thumbnail_url}
          draggable={false}
          w={10}
          h={10}
          borderRadius="base"
          borderBottomRadius="lg"
          objectFit="cover"
        />
        <Flex flexDir="column" gap={1} flex={1}>
          <Text className="picker-option">{board_name}</Text>
        </Flex>
      </Flex>
    );
  }
);

BoardsSwitcherOptionComponent.displayName = 'BoardsSwitcherOptionComponent';
