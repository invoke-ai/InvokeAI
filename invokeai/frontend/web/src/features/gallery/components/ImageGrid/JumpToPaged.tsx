import {
  Button,
  Flex,
  NumberInput,
  NumberInputField,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

type JumpToPagedProps = {
  pageIndex: number;
  pageCount: number;
  onChange: (valueAsString: string, valueAsNumber: number) => void;
};

export const JumpToPaged = memo(({ pageIndex, pageCount, onChange }: JumpToPagedProps) => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();
  const [newPageInput, setNewPageInput] = useState(String(pageIndex + 1));
  const inputWrapperRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (disclosure.isOpen) {
      const nextPage = pageIndex + 1;
      setNewPageInput(String(nextPage));
      setTimeout(() => {
        const input = inputWrapperRef.current?.querySelector('input');
        input?.focus();
        input?.select();
      }, 0);
    }
  }, [disclosure.isOpen, pageIndex]);

  const onChangeJumpTo = useCallback((valueAsString: string, _valueAsNumber: number) => {
    setNewPageInput(valueAsString);
  }, []);

  const onClickGo = useCallback(() => {
    if (!newPageInput) {
      return;
    }
    const parsed = Number.parseInt(newPageInput, 10);
    if (Number.isNaN(parsed)) {
      return;
    }
    onChange(String(parsed), parsed);
    disclosure.onClose();
  }, [disclosure, newPageInput, onChange]);

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
      const nextPage = pageIndex + 1;
      setNewPageInput(String(nextPage));
      disclosure.onClose();
    },
    { enabled: disclosure.isOpen, enableOnFormTags: ['input'] },
    [disclosure.isOpen, pageIndex, disclosure.onClose]
  );

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
          <Flex gap={2} alignItems="center">
            <NumberInput
              ref={inputWrapperRef}
              min={1}
              max={pageCount}
              value={newPageInput}
              onChange={onChangeJumpTo}
              size="sm"
              w="72px"
              clampValueOnBlur
            >
              <NumberInputField title="" textAlign="center" />
            </NumberInput>
            <Button h="full" size="sm" onClick={onClickGo}>
              {t('gallery.go')}
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

JumpToPaged.displayName = 'JumpToPaged';
