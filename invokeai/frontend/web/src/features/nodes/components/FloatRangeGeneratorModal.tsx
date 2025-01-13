import {
  Button,
  CompositeNumberInput,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { round } from 'lodash-es';
import { atom } from 'nanostores';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

type FloatRangeGeneratorModalState = {
  isOpen: boolean;
  onSave: (values: number[]) => void;
};

const $floatRangeGeneratorModal = atom<FloatRangeGeneratorModalState>({
  isOpen: false,
  onSave: () => {},
});

export const openFloatRangeGeneratorModal = (onSave: (values: number[]) => void) => {
  $floatRangeGeneratorModal.set({ ...$floatRangeGeneratorModal.get(), isOpen: true, onSave });
};

const onClose = () => {
  $floatRangeGeneratorModal.set({ ...$floatRangeGeneratorModal.get(), isOpen: false });
};

export const FloatRangeGeneratorModal = memo(() => {
  const { isOpen, onSave } = useStore($floatRangeGeneratorModal);
  const { t } = useTranslation();

  const [start, setStart] = useState(0);
  const [step, setStep] = useState(1);
  const [count, setCount] = useState(1);

  const values = useMemo(() => Array.from({ length: count }, (_, i) => start + i * step), [start, step, count]);

  const onReset = useCallback(() => {
    setStart(0);
    setStep(1);
    setCount(1);
  }, []);

  const onClickSave = useCallback(() => {
    onSave(values);
    onClose();
  }, [onSave, values]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{t('nodes.floatRangeGenerator')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody minH={200}>
          <Flex gap={4} alignItems="flex-end">
            <FormControl orientation="vertical">
              <FormLabel>{t('common.start')}</FormLabel>
              <CompositeNumberInput value={start} onChange={setStart} min={-Infinity} max={Infinity} step={0.01} />
            </FormControl>
            <FormControl orientation="vertical">
              <FormLabel>{t('common.count')}</FormLabel>
              <CompositeNumberInput value={count} onChange={setCount} min={1} max={Infinity} />
            </FormControl>
            <FormControl orientation="vertical">
              <FormLabel>{t('common.step')}</FormLabel>
              <CompositeNumberInput value={step} onChange={setStep} min={-Infinity} max={Infinity} step={0.01} />
            </FormControl>
            <IconButton aria-label="Reset" icon={<PiArrowCounterClockwiseBold />} onClick={onReset} variant="ghost" />
          </Flex>
          <Flex w="full" h="auto" flexDir="column" gap={2} pt={4}>
            <FormLabel>{t('common.values')}</FormLabel>
            <Flex w="full" h="full" p={2} borderWidth={1} borderRadius="base">
              <Text fontFamily="monospace" fontSize="md" color="base.300">
                {values.map((val) => round(val, 2)).join(', ')}
              </Text>
            </Flex>
          </Flex>
        </ModalBody>
        <ModalFooter gap={2}>
          <Button onClick={onClose} variant="ghost">
            {t('common.cancel')}
          </Button>
          <Button onClick={onClickSave}>{t('common.save')}</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
});

FloatRangeGeneratorModal.displayName = 'FloatRangeGeneratorModal';
