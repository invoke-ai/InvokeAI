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
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type JumpToPagedProps = {
  pageIndex: number;
  pageCount: number;
  onChange: (valueAsString: string, valueAsNumber: number) => void;
};

export const JumpToPaged = memo(({ pageIndex, pageCount, onChange }: JumpToPagedProps) => {
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
          <Flex gap={2} alignItems="center">
            <NumberInput
              min={1}
              max={pageCount}
              value={pageIndex + 1}
              onChange={onChange}
              size="sm"
              w="72px"
              clampValueOnBlur
            >
              <NumberInputField title="" textAlign="center" />
            </NumberInput>
            <Button h="full" size="sm" onClick={disclosure.onClose}>
              {t('gallery.go')}
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

JumpToPaged.displayName = 'JumpToPaged';
