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
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

type JumpToPagedProps = {
  pageIndex: number;
  pageCount: number;
  onChange: (valueAsString: string, valueAsNumber: number) => void;
};

export const JumpToPaged = memo(({ pageIndex, pageCount, onChange }: JumpToPagedProps) => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();
  const [newPage, setNewPage] = useState(pageIndex + 1);

  useEffect(() => {
    if (disclosure.isOpen) {
      setNewPage(pageIndex + 1);
    }
  }, [disclosure.isOpen, pageIndex]);

  const onChangeJumpTo = useCallback((valueAsString: string, valueAsNumber: number) => {
    if (!valueAsString) {
      return;
    }
    if (Number.isNaN(valueAsNumber)) {
      return;
    }
    setNewPage(valueAsNumber);
  }, []);

  const onClickGo = useCallback(() => {
    onChange(String(newPage), newPage);
    disclosure.onClose();
  }, [disclosure, newPage, onChange]);

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
              min={1}
              max={pageCount}
              value={newPage}
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
