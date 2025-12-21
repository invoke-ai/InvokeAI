import {
  Button,
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
import { useAppDispatch } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { Hotkey } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useHotkeyData } from 'features/system/components/HotkeysModal/useHotkeyData';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import { allHotkeysReset } from 'features/system/store/hotkeysSlice';
import type { ChangeEventHandler, ReactElement } from 'react';
import { cloneElement, Fragment, memo, useCallback, useMemo, useState } from 'react';
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
  const dispatch = useAppDispatch();
  const [hotkeyFilter, setHotkeyFilter] = useState('');
  const [isEditMode, setIsEditMode] = useState(false);
  const clearHotkeyFilter = useCallback(() => setHotkeyFilter(''), []);
  const onChange = useCallback<ChangeEventHandler<HTMLInputElement>>((e) => setHotkeyFilter(e.target.value), []);
  const toggleEditMode = useCallback(() => setIsEditMode((prev) => !prev), []);
  const handleResetAll = useCallback(() => {
    dispatch(allHotkeysReset());
  }, [dispatch]);
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
        if (!item.isEnabled) {
          return;
        }
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
      <Modal isOpen={isOpen} onClose={onClose} isCentered size="2xl" useInert={false}>
        <ModalOverlay />
        <ModalContent maxH="80vh" h="80vh">
          <ModalHeader>{t('hotkeys.hotkeys')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody display="flex" flexDir="column" gap={4}>
            <InputGroup>
              <Input autoFocus placeholder={t('hotkeys.searchHotkeys')} value={hotkeyFilter} onChange={onChange} tabIndex={1} />
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
                        <HotkeyListItem hotkey={hotkey} showEditor={isEditMode} />
                        {i < category.hotkeys.length - 1 && <Divider />}
                      </Fragment>
                    ))}
                  </StickyScrollable>
                ))}
                {!filteredHotkeys.length && <IAINoContentFallback label={t('hotkeys.noHotkeysFound')} icon={null} />}
              </Flex>
            </ScrollableContent>
          </ModalBody>
          <ModalFooter>
            <Flex gap={2} w="full" justifyContent="space-between">
              <Button onClick={toggleEditMode} size="sm">
                {isEditMode ? t('hotkeys.viewMode') : t('hotkeys.editMode')}
              </Button>
              {isEditMode && (
                <Button onClick={handleResetAll} size="sm" colorScheme="error">
                  {t('hotkeys.resetAll')}
                </Button>
              )}
            </Flex>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(HotkeysModal);
