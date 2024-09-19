import {
  Flex,
  Image,
  Popover,
  PopoverBody,
  PopoverCloseButton,
  PopoverContent,
  PopoverHeader,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { CanvasV2Announcement } from 'features/notifications/components/canvasV2Announcement';
import { NotificationIcon } from 'features/notifications/components/NotificationIcon';
import InvokeSymbol from 'public/assets/images/invoke-favicon.png';
import { useTranslation } from 'react-i18next';

export const Notifications = () => {
  const { t } = useTranslation();
  return (
    <Popover>
      <PopoverTrigger>
        <NotificationIcon showIndicator={true} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverCloseButton />
        <PopoverHeader fontSize="md" fontWeight="semibold">
          <Flex alignItems="center" gap={3}>
            <Image src={InvokeSymbol} boxSize={6} />
            {t('whatsNew.whatsNewInInvoke')}
          </Flex>
        </PopoverHeader>
        <PopoverBody p={2}>
          <CanvasV2Announcement />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
