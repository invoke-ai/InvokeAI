import { Flex, Text } from '@invoke-ai/ui-library';
import { IconSwitch } from 'common/components/IconSwitch';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiPencilBold } from 'react-icons/pi';

const TooltipEdit = memo(() => {
  const { t } = useTranslation();

  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">{t('common.edit')}</Text>
      <Text fontWeight="normal">{t('common.editDesc')}</Text>
    </Flex>
  );
});
TooltipEdit.displayName = 'TooltipEdit';

const TooltipView = memo(() => {
  const { t } = useTranslation();

  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">{t('common.view')}</Text>
      <Text fontWeight="normal">{t('common.viewDesc')}</Text>
    </Flex>
  );
});
TooltipView.displayName = 'TooltipView';

export const ViewerToggle = memo(() => {
  const imageViewer = useImageViewer();
  useHotkeys('z', imageViewer.onToggle, [imageViewer]);
  useHotkeys('esc', imageViewer.onClose, [imageViewer]);
  const onChange = useCallback(
    (isChecked: boolean) => {
      if (isChecked) {
        imageViewer.onClose();
      } else {
        imageViewer.onOpen();
      }
    },
    [imageViewer]
  );

  return (
    <IconSwitch
      isChecked={!imageViewer.isOpen}
      onChange={onChange}
      iconUnchecked={<PiEyeBold />}
      tooltipUnchecked={<TooltipView />}
      iconChecked={<PiPencilBold />}
      tooltipChecked={<TooltipEdit />}
      ariaLabel="Toggle viewer"
    />
  );
});

ViewerToggle.displayName = 'ViewerToggle';
