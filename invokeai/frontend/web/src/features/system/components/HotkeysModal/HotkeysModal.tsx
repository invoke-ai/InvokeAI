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
import type { HotkeyGroup } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useHotkeyData } from 'features/system/components/HotkeysModal/useHotkeyData';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import type { ChangeEventHandler, ReactElement } from 'react';
import { cloneElement, Fragment, memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

import HotkeyListItem from './HotkeyListItem';

type HotkeysModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

const HotkeysModal = ({ children }: HotkeysModalProps) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();
  const [hotkeyFilter, setHotkeyFilter] = useState('');
  const clearHotkeyFilter = useCallback(() => setHotkeyFilter(''), []);
  const onChange = useCallback<ChangeEventHandler<HTMLInputElement>>((e) => setHotkeyFilter(e.target.value), []);
  const hotkeyGroups = useHotkeyData();
  const filteredHotkeyGroups = useMemo(() => {
    if (!hotkeyFilter.trim().length) {
      return hotkeyGroups;
    }
    const trimmedHotkeyFilter = hotkeyFilter.trim().toLowerCase();
    const filteredGroups: HotkeyGroup[] = [];
    hotkeyGroups.forEach((group) => {
      const filteredGroup: HotkeyGroup = {
        title: group.title,
        hotkeyListItems: [],
      };
      group.hotkeyListItems.forEach((item) => {
        if (
          item.title.toLowerCase().includes(trimmedHotkeyFilter) ||
          item.desc.toLowerCase().includes(trimmedHotkeyFilter) ||
          item.hotkeys.some((hotkey) => hotkey.some((key) => key.toLowerCase().includes(trimmedHotkeyFilter)))
        ) {
          filteredGroup.hotkeyListItems.push(item);
        }
      });
      if (filteredGroup.hotkeyListItems.length) {
        filteredGroups.push(filteredGroup);
      }
    });
    return filteredGroups;
  }, [hotkeyGroups, hotkeyFilter]);

  return (
    <>
      {cloneElement(children, {
        onClick: onOpen,
      })}
      <Modal isOpen={isOpen} onClose={onClose} isCentered size="2xl">
        <ModalOverlay />
        <ModalContent maxH="80vh" h="80vh">
          <ModalHeader>{t('hotkeys.keyboardShortcuts')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody display="flex" flexDir="column" gap={4}>
            <InputGroup>
              <Input placeholder={t('hotkeys.searchHotkeys')} value={hotkeyFilter} onChange={onChange} />
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
                {filteredHotkeyGroups.map((group) => (
                  <StickyScrollable key={group.title} title={group.title}>
                    {group.hotkeyListItems.map((hotkey, i) => (
                      <Fragment key={i}>
                        <HotkeyListItem title={hotkey.title} description={hotkey.desc} hotkeys={hotkey.hotkeys} />
                        {i < group.hotkeyListItems.length - 1 && <Divider />}
                      </Fragment>
                    ))}
                  </StickyScrollable>
                ))}
                {!filteredHotkeyGroups.length && (
                  <IAINoContentFallback label={t('hotkeys.noHotkeysFound')} icon={null} />
                )}
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
