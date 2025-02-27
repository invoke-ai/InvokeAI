import {
  Box,
  Divider,
  Flex,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import BoardAutoAddSelect from 'features/gallery/components/Boards/BoardAutoAddSelect';
import AutoAssignBoardCheckbox from 'features/gallery/components/GallerySettingsPopover/AutoAssignBoardCheckbox';
import ShowArchivedBoardsCheckbox from 'features/gallery/components/GallerySettingsPopover/ShowArchivedBoardsCheckbox';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixFill } from 'react-icons/pi';

import { BoardsListSortControls } from './BoardsListSortControls';

const BoardsSettingsPopover = () => {
  const { t } = useTranslation();

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          size="sm"
          variant="link"
          alignSelf="stretch"
          aria-label={t('gallery.boardsSettings')}
          icon={<PiGearSixFill />}
          tooltip={t('gallery.boardsSettings')}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <AutoAssignBoardCheckbox />
            <ShowArchivedBoardsCheckbox />
            <BoardAutoAddSelect />
            <Box py={2}>
              <Divider />
            </Box>

            <BoardsListSortControls />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(BoardsSettingsPopover);
