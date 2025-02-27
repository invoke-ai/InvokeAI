import {
  Divider,
  Flex,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { Hotkey } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useHotkeyData } from 'features/system/components/HotkeysModal/useHotkeyData';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import type { ChangeEventHandler, ReactElement } from 'react';
import { cloneElement, Fragment, memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

import HotkeyListItem from './HotkeyListItem';

type HotkeysModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

type TransformedHotkeysCategoryData = {
  title: string;
  hotkeys: Hotkey[];
};

const HotkeysModal = ({ children }: HotkeysModalProps) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();
  const [hotkeyFilter, setHotkeyFilter] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);
  const clearHotkeyFilter = useCallback(() => setHotkeyFilter(''), []);
  const onChange = useCallback<ChangeEventHandler<HTMLInputElement>>((e) => setHotkeyFilter(e.target.value), []);
  const hotkeysData = useHotkeyData();
  const filteredHotkeys = useMemo<TransformedHotkeysCategoryData[]>(() => {
    const trimmedHotkeyFilter = hotkeyFilter.trim().toLowerCase();
    const filteredCategories: TransformedHotkeysCategoryData[] = [];
    Object.values(hotkeysData).forEach((category) => {
      const filteredGroup: TransformedHotkeysCategoryData = {
        title: category.title,
        hotkeys: [],
      };
      Object.values(category.hotkeys).forEach((item) => {
        if (!trimmedHotkeyFilter.length) {
          filteredGroup.hotkeys.push(item);
        } else if (item.title.toLowerCase().includes(trimmedHotkeyFilter)) {
          filteredGroup.hotkeys.push(item);
        } else if (item.desc.toLowerCase().includes(trimmedHotkeyFilter)) {
          filteredGroup.hotkeys.push(item);
        } else if (item.category.toLowerCase().includes(trimmedHotkeyFilter)) {
          filteredGroup.hotkeys.push(item);
        } else if (
          item.platformKeys.some((hotkey) => hotkey.some((key) => key.toLowerCase().includes(trimmedHotkeyFilter)))
        ) {
          filteredGroup.hotkeys.push(item);
        }
      });
      if (filteredGroup.hotkeys.length) {
        filteredCategories.push(filteredGroup);
      }
    });
    return filteredCategories;
  }, [hotkeysData, hotkeyFilter]);

  return (
    <>
      {cloneElement(children, {
        onClick: onOpen,
      })}
      <Modal isOpen={isOpen} onClose={onClose} isCentered size="2xl" useInert={false} initialFocusRef={inputRef}>
        <ModalOverlay />
        <ModalContent maxH="80vh" h="80vh">
          <ModalHeader>{t('hotkeys.hotkeys')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody display="flex" flexDir="column" gap={4}>
            <InputGroup>
              <Input ref={inputRef} placeholder={t('hotkeys.searchHotkeys')} value={hotkeyFilter} onChange={onChange} />
              {hotkeyFilter.length && (
                <InputRightElement h="full" pe={2}>
                  <IconButton
                    onClick={clearHotkeyFilter}
                    size="sm"
                    variant="ghost"
                    aria-label={t('hotkeys.clearSearch')}
                    boxSize={4}
                    icon={<PiXBold />}
                  />
                </InputRightElement>
              )}
            </InputGroup>

            <ScrollableContent>
              <Flex flexDir="column" gap={4}>
                {filteredHotkeys.map((category) => (
                  <StickyScrollable key={category.title} title={category.title}>
                    {category.hotkeys.map((hotkey, i) => (
                      <Fragment key={hotkey.id}>
                        <HotkeyListItem hotkey={hotkey} />
                        {i < category.hotkeys.length - 1 && <Divider />}
                      </Fragment>
                    ))}
                  </StickyScrollable>
                ))}
                {!filteredHotkeys.length && <IAINoContentFallback label={t('hotkeys.noHotkeysFound')} icon={null} />}
              </Flex>
            </ScrollableContent>
          </ModalBody>
          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(HotkeysModal);
