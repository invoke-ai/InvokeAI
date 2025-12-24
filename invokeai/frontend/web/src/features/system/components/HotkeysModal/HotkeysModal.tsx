import {
  Button,
  ConfirmationAlertDialog,
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
  Spinner,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { Hotkey } from 'features/system/components/HotkeysModal/useHotkeyData';
import {
  isHotkeysModified,
  useHotkeyConflictMap,
  useHotkeyData,
} from 'features/system/components/HotkeysModal/useHotkeyData';
import { allHotkeysReset } from 'features/system/store/hotkeysSlice';
import type { ChangeEventHandler, ReactElement } from 'react';
import { cloneElement, memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

import { HotkeysListWrapper } from './HotkeysListWrapper';

type HotkeysModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

type TransformedHotkeysCategoryData = {
  title: string;
  hotkeys: Hotkey[];
};

// Helper to check if a hotkey matches the search filter
const matchesFilter = (item: Hotkey, filter: string): boolean => {
  return [item.title, item.desc, item.category, ...item.platformKeys.flat()].some((text) =>
    text.toLowerCase().includes(filter)
  );
};

const HotkeysModalInner = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { isOpen: isResetDialogOpen, onOpen: onResetDialogOpen, onClose: onResetDialogClose } = useDisclosure();
  const [hotkeyFilter, setHotkeyFilter] = useState('');
  const clearHotkeyFilter = useCallback(() => setHotkeyFilter(''), []);
  const onChange = useCallback<ChangeEventHandler<HTMLInputElement>>((e) => setHotkeyFilter(e.target.value), []);
  const handleResetAll = useCallback(() => {
    dispatch(allHotkeysReset());
    onResetDialogClose();
  }, [dispatch, onResetDialogClose]);

  const hotkeysData = useHotkeyData();
  const conflictMap = useHotkeyConflictMap();

  const filteredHotkeys = useMemo<TransformedHotkeysCategoryData[]>(() => {
    const trimmedHotkeyFilter = hotkeyFilter.trim().toLowerCase();
    const filteredCategories: TransformedHotkeysCategoryData[] = [];
    for (const category of Object.values(hotkeysData)) {
      const filteredGroup: TransformedHotkeysCategoryData = {
        title: category.title,
        hotkeys: [],
      };
      for (const item of Object.values(category.hotkeys)) {
        if (!item.isEnabled) {
          continue;
        }
        if (!trimmedHotkeyFilter.length || matchesFilter(item, trimmedHotkeyFilter)) {
          filteredGroup.hotkeys.push(item);
        }
      }
      if (filteredGroup.hotkeys.length) {
        filteredCategories.push(filteredGroup);
      }
    }
    return filteredCategories;
  }, [hotkeysData, hotkeyFilter]);

  const canResetHotkeys = useMemo(() => isHotkeysModified(hotkeysData), [hotkeysData]);

  return (
    <>
      <ModalBody display="flex" flexDir="column" gap={4}>
        <InputGroup>
          <Input
            autoFocus
            placeholder={t('hotkeys.searchHotkeys')}
            value={hotkeyFilter}
            onChange={onChange}
            tabIndex={1}
          />
          {hotkeyFilter.length > 0 ? (
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
          ) : null}
        </InputGroup>

        <ScrollableContent>
          <Flex flexDir="column">
            {filteredHotkeys.map((category) => (
              <HotkeysListWrapper
                key={category.title}
                title={category.title}
                hotkeysList={category.hotkeys}
                conflictMap={conflictMap}
                t={t}
                dispatch={dispatch}
              />
            ))}
            {!filteredHotkeys.length && <IAINoContentFallback label={t('hotkeys.noHotkeysFound')} icon={null} />}
          </Flex>
        </ScrollableContent>
      </ModalBody>

      <ModalFooter>
        <Flex gap={2} w="full" justifyContent="space-between">
          <Button onClick={onResetDialogOpen} size="sm" colorScheme="error" disabled={!canResetHotkeys}>
            {t('hotkeys.resetAll')}
          </Button>
        </Flex>
      </ModalFooter>

      <ConfirmationAlertDialog
        isOpen={isResetDialogOpen}
        onClose={onResetDialogClose}
        title={t('hotkeys.resetAll')}
        acceptCallback={handleResetAll}
        acceptButtonText={t('common.reset')}
        useInert={false}
      >
        <Flex flexDirection="column" gap={2}>
          <Text>{t('hotkeys.resetAllConfirmation')}</Text>
        </Flex>
      </ConfirmationAlertDialog>
    </>
  );
});

HotkeysModalInner.displayName = 'HotkeysModalInner';

const HotkeysModalContent = memo(() => {
  const { t } = useTranslation();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const timeout = setTimeout(() => setIsReady(true), 0);
    return () => clearTimeout(timeout);
  }, []);

  return (
    <>
      <ModalHeader>{t('hotkeys.hotkeys')}</ModalHeader>
      <ModalCloseButton />
      {isReady ? (
        <HotkeysModalInner />
      ) : (
        <ModalBody display="flex" flexDir="column" gap={4}>
          <Flex flex="1" alignItems="center" justifyContent="center">
            <Spinner />
          </Flex>
        </ModalBody>
      )}
    </>
  );
});

HotkeysModalContent.displayName = 'HotkeysModalContent';

const HotkeysModal = ({ children }: HotkeysModalProps) => {
  const { isOpen, onOpen, onClose } = useDisclosure();

  return (
    <>
      {cloneElement(children, {
        onClick: onOpen,
      })}
      <Modal isOpen={isOpen} onClose={onClose} isCentered size="2xl" useInert={false}>
        <ModalOverlay />
        <ModalContent h="80vh">
          {isOpen && <HotkeysModalContent />}
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(HotkeysModal);
