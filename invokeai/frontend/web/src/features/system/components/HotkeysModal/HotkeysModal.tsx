import { CloseIcon } from '@chakra-ui/icons';
import {
  Divider,
  Flex,
  InputGroup,
  InputRightElement,
  useDisclosure,
} from '@chakra-ui/react';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvInput } from 'common/components/InvInput/InvInput';
import {
  InvModal,
  InvModalBody,
  InvModalCloseButton,
  InvModalContent,
  InvModalFooter,
  InvModalHeader,
  InvModalOverlay,
} from 'common/components/InvModal/wrapper';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { HotkeyGroup } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useHotkeyData } from 'features/system/components/HotkeysModal/useHotkeyData';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import type { ChangeEventHandler, ReactElement } from 'react';
import {
  cloneElement,
  Fragment,
  memo,
  useCallback,
  useMemo,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';

import HotkeyListItem from './HotkeyListItem';

type HotkeysModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

const HotkeysModal = ({ children }: HotkeysModalProps) => {
  const {
    isOpen: isHotkeyModalOpen,
    onOpen: onHotkeysModalOpen,
    onClose: onHotkeysModalClose,
  } = useDisclosure();
  const { t } = useTranslation();
  const [hotkeyFilter, setHotkeyFilter] = useState('');
  const clearHotkeyFilter = useCallback(() => setHotkeyFilter(''), []);
  const onChange = useCallback<ChangeEventHandler<HTMLInputElement>>(
    (e) => setHotkeyFilter(e.target.value),
    []
  );
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
          item.hotkeys.some((hotkey) =>
            hotkey.some((key) =>
              key.toLowerCase().includes(trimmedHotkeyFilter)
            )
          )
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
        onClick: onHotkeysModalOpen,
      })}
      <InvModal
        isOpen={isHotkeyModalOpen}
        onClose={onHotkeysModalClose}
        isCentered
        size="2xl"
      >
        <InvModalOverlay />
        <InvModalContent maxH="80vh" h="80vh">
          <InvModalHeader>{t('hotkeys.keyboardShortcuts')}</InvModalHeader>
          <InvModalCloseButton />
          <InvModalBody display="flex" flexDir="column" gap={4}>
            <InputGroup>
              <InvInput
                placeholder={t('hotkeys.searchHotkeys')}
                value={hotkeyFilter}
                onChange={onChange}
              />
              {hotkeyFilter.length && (
                <InputRightElement h="full" pe={2}>
                  <InvIconButton
                    onClick={clearHotkeyFilter}
                    size="sm"
                    variant="ghost"
                    aria-label={t('hotkeys.clearSearch')}
                    icon={<CloseIcon boxSize={3} />}
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
                        <HotkeyListItem
                          title={hotkey.title}
                          description={hotkey.desc}
                          hotkeys={hotkey.hotkeys}
                        />
                        {i < group.hotkeyListItems.length - 1 && <Divider />}
                      </Fragment>
                    ))}
                  </StickyScrollable>
                ))}
                {!filteredHotkeyGroups.length && (
                  <IAINoContentFallback
                    label={t('hotkeys.noHotkeysFound')}
                    icon={null}
                  />
                )}
              </Flex>
            </ScrollableContent>
          </InvModalBody>
          <InvModalFooter />
        </InvModalContent>
      </InvModal>
    </>
  );
};

export default memo(HotkeysModal);
