import { CloseIcon } from '@chakra-ui/icons';
import {
  Divider,
  Flex,
  InputGroup,
  InputRightElement,
  useDisclosure,
} from '@chakra-ui/react';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InvHeading } from 'common/components/InvHeading/wrapper';
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
              {filteredHotkeyGroups.map((group) => (
                <Flex key={group.title} pb={4} flexDir="column">
                  <Flex
                    ps={2}
                    pb={4}
                    position="sticky"
                    zIndex={1}
                    top={0}
                    bg="base.800"
                  >
                    <InvHeading size="sm">{group.title}</InvHeading>
                  </Flex>
                  <Flex
                    key={group.title}
                    p={4}
                    borderRadius="base"
                    bg="base.750"
                    flexDir="column"
                    gap={4}
                  >
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
                  </Flex>
                </Flex>
              ))}
              {!filteredHotkeyGroups.length && (
                <IAINoContentFallback
                  label={t('hotkeys.noHotkeysFound')}
                  icon={null}
                />
              )}
            </ScrollableContent>
          </InvModalBody>
          <InvModalFooter />
        </InvModalContent>
      </InvModal>
    </>
  );
};

export default memo(HotkeysModal);
