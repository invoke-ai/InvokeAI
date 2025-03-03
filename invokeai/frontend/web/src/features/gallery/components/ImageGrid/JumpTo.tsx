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
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const JumpTo = memo(() => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();

  return (
    <Popover isOpen={disclosure.isOpen} onClose={disclosure.onClose} isLazy lazyBehavior="unmount">
      <PopoverTrigger>
        <Button aria-label={t('gallery.jump')} size="sm" onClick={disclosure.onToggle} variant="outline">
          {t('gallery.jump')}
        </Button>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <JumpToContent disclosure={disclosure} />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

JumpTo.displayName = 'JumpTo';

const JumpToContent = memo(({ disclosure }: { disclosure: ReturnType<typeof useDisclosure> }) => {
  const { t } = useTranslation();
  const { goToPage, currentPage, pages } = useGalleryPagination();
  const [newPage, setNewPage] = useState(currentPage);
  const ref = useRef<HTMLInputElement>(null);

  const onChangeJumpTo = useCallback((v: number) => {
    setNewPage(v - 1);
  }, []);

  const onClickGo = useCallback(() => {
    goToPage(newPage);
    disclosure.onClose();
  }, [goToPage, newPage, disclosure]);

  useHotkeys(
    'enter',
    () => {
      onClickGo();
    },
    { enabled: disclosure.isOpen, enableOnFormTags: ['input'] },
    [disclosure.isOpen, onClickGo]
  );

  useHotkeys(
    'esc',
    () => {
      setNewPage(currentPage);
      disclosure.onClose();
    },
    { enabled: disclosure.isOpen, enableOnFormTags: ['input'] },
    [disclosure.isOpen, disclosure.onClose]
  );

  useEffect(() => {
    setTimeout(() => {
      const input = ref.current?.querySelector('input');
      input?.focus();
      input?.select();
    }, 0);
    setNewPage(currentPage);
  }, [currentPage]);

  return (
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
      <Button h="full" size="sm" onClick={onClickGo}>
        {t('gallery.go')}
      </Button>
    </Flex>
  );
});
JumpToContent.displayName = 'JumpToContent';
