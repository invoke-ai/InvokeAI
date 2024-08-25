import { Button, Flex, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiPencilBold } from 'react-icons/pi';

export const ViewerToggleMenu = () => {
  const { t } = useTranslation();
  const { isOpen, onToggle, onClose, onOpen } = useImageViewer();

  const handleOpen = useCallback(() => onOpen(), [onOpen]);
  const handleClose = useCallback(() => onClose(), [onClose]);

  useHotkeys('z', onToggle, [onToggle]);
  useHotkeys('esc', onClose, [onClose]);

  return (
    <Flex gap={4} alignItems="center" justifyContent="center">
      <Text>{isOpen ? t('common.viewing') : t('common.editing')}</Text>
      <Flex gap={1} alignItems="center" borderWidth={1} borderRadius="md" padding={1}>
        <Tooltip
          hasArrow
          label={
            <Flex flexDir="column">
              <Text fontWeight="semibold">{t('common.viewing')}</Text>
              <Text fontWeight="normal">{t('common.viewingDesc')}</Text>
            </Flex>
          }
        >
          <Button
            onClick={handleOpen}
            variant={isOpen ? 'solid' : 'ghost'}
            colorScheme="invokeBlue"
            size="sm"
            aria-label={t('common.viewing')}
          >
            <Icon as={PiEyeBold} boxSize="0.95rem" />
          </Button>
        </Tooltip>
        <Tooltip
          hasArrow
          label={
            <Flex flexDir="column">
              <Text fontWeight="semibold">{t('common.editing')}</Text>
              <Text fontWeight="normal">{t('common.editingDesc')}</Text>
            </Flex>
          }
        >
          <Button
            onClick={handleClose}
            variant={!isOpen ? 'solid' : 'ghost'}
            colorScheme="invokeBlue"
            size="sm"
            aria-label={t('common.editing')}
          >
            <Icon as={PiPencilBold} boxSize="0.95rem" />
          </Button>
        </Tooltip>
      </Flex>
    </Flex>
  );
};
