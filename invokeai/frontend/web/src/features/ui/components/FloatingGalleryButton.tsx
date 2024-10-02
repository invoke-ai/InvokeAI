import { Flex, IconButton, Tooltip } from '@invoke-ai/ui-library';
import type { UsePanelReturn } from 'features/ui/hooks/usePanel';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesSquareBold } from 'react-icons/pi';

type Props = {
  panelApi: UsePanelReturn;
};

const FloatingGalleryButton = (props: Props) => {
  const { t } = useTranslation();

  return (
    <Flex pos="absolute" transform="translate(0, -50%)" minW={8} top="50%" insetInlineEnd={2} zIndex={11}>
      <Tooltip label={t('accessibility.showGalleryPanel')} placement="start">
        <IconButton
          aria-label={t('accessibility.showGalleryPanel')}
          onClick={props.panelApi.toggle}
          icon={<PiImagesSquareBold />}
          h={48}
        />
      </Tooltip>
    </Flex>
  );
};

export default memo(FloatingGalleryButton);
