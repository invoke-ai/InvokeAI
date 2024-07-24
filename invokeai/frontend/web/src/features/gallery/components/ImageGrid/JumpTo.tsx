import {
  Button,
  CompositeNumberInput,
  Flex,
  FormControl,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { useCallback, useEffect, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const JumpTo = () => {
  const { t } = useTranslation();
  const { goToPage, currentPage, pages } = useGalleryPagination();
  const [newPage, setNewPage] = useState(currentPage);
  const { isOpen, onToggle, onClose } = useDisclosure();

  const onChangeJumpTo = useCallback((v: number) => {
    setNewPage(v - 1);
  }, []);

  const onClickGo = useCallback(() => {
    goToPage(newPage);
    onClose();
  }, [newPage, goToPage, onClose]);

  useHotkeys(
    'enter',
    () => {
      if (isOpen) {
        onClickGo();
      }
    },
    [isOpen, onClickGo]
  );

  useEffect(() => {
    setNewPage(currentPage);
  }, [currentPage]);

  return (
    <Popover isOpen={isOpen} onClose={onClose}>
      <PopoverTrigger>
        <Button aria-label={t('gallery.jump')} size="xs" onClick={onToggle}>
          {t('gallery.jump')}
        </Button>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <Flex gap={2}>
            <FormControl>
              <CompositeNumberInput
                size="sm"
                maxW="60px"
                value={newPage + 1}
                min={1}
                max={pages}
                step={1}
                onChange={onChangeJumpTo}
              />
              <Button size="sm" onClick={onClickGo}>
                {t('gallery.go')}
              </Button>
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
