import { Flex } from '@chakra-ui/layout';
import { Portal } from '@chakra-ui/portal';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import type { UsePanelReturn } from 'features/ui/hooks/usePanel';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesSquareBold } from 'react-icons/pi'

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
      <Flex
        pos="absolute"
        transform="translate(0, -50%)"
        minW={8}
        top="50%"
        insetInlineEnd={0}
      >
        <InvTooltip
          label={t('accessibility.showGalleryPanel')}
          placement="start"
        >
          <InvIconButton
            aria-label={t('accessibility.showGalleryPanel')}
            onClick={props.panelApi.expand}
            icon={<PiImagesSquareBold size="20px" />}
            p={0}
            h={48}
            borderEndRadius={0}
          />
        </InvTooltip>
      </Flex>
    </Portal>
  );
};

export default memo(FloatingGalleryButton);
