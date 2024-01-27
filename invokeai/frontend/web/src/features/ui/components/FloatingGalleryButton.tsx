import { Flex, IconButton, Portal, Tooltip } from '@invoke-ai/ui-library';
import type { UsePanelReturn } from 'features/ui/hooks/usePanel';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesSquareBold } from 'react-icons/pi';

type Props = {
  panelApi: UsePanelReturn;
};

const FloatingGalleryButton = (props: Props) => {
  const { t } = useTranslation();

  if (!props.panelApi.isCollapsed) {
    return null;
  }

  return (
    <Portal>
      <Flex pos="absolute" transform="translate(0, -50%)" minW={8} top="50%" insetInlineEnd="21px">
        <Tooltip label={t('accessibility.showGalleryPanel')} placement="start">
          <IconButton
            aria-label={t('accessibility.showGalleryPanel')}
            onClick={props.panelApi.expand}
            icon={<PiImagesSquareBold size="20px" />}
            p={0}
            h={48}
            borderEndRadius={0}
          />
        </Tooltip>
      </Flex>
    </Portal>
  );
};

export default memo(FloatingGalleryButton);
