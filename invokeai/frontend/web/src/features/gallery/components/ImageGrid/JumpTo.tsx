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
import { useCallback, useEffect, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const JumpTo = () => {
  const { t } = useTranslation();
  const { goToPage, currentPage, pages } = useGalleryPagination();
  const [newPage, setNewPage] = useState(currentPage);
  const { isOpen, onToggle, onClose } = useDisclosure();
  const ref = useRef<HTMLInputElement>(null);

  const onOpen = useCallback(() => {
    setNewPage(currentPage);
    setTimeout(() => {
      const input = ref.current?.querySelector('input');
      input?.focus();
      input?.select();
    }, 0);
  }, [currentPage]);

  const onChangeJumpTo = useCallback((v: number) => {
    setNewPage(v - 1);
  }, []);

  const onPointerUpGo = useCallback(() => {
    goToPage(newPage);
    onClose();
  }, [newPage, goToPage, onClose]);

  useHotkeys(
    'enter',
    () => {
      onPointerUpGo();
    },
    { enabled: isOpen, enableOnFormTags: ['input'] },
    [isOpen, onPointerUpGo]
  );

  useHotkeys(
    'esc',
    () => {
      setNewPage(currentPage);
      onClose();
    },
    { enabled: isOpen, enableOnFormTags: ['input'] },
    [isOpen, onClose]
  );

  useEffect(() => {
    setNewPage(currentPage);
  }, [currentPage]);

  return (
    <Popover isOpen={isOpen} onClose={onClose} onOpen={onOpen}>
      <PopoverTrigger>
        <Button aria-label={t('gallery.jump')} size="sm" onPointerUp={onToggle} variant="outline">
          {t('gallery.jump')}
        </Button>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <Flex gap={2} alignItems="center">
            <FormControl>
              <CompositeNumberInput
                ref={ref}
                size="sm"
                maxW="60px"
                value={newPage + 1}
                min={1}
                max={pages}
                step={1}
                onChange={onChangeJumpTo}
              />
            </FormControl>
            <Button h="full" size="sm" onPointerUp={onPointerUpGo}>
              {t('gallery.go')}
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
