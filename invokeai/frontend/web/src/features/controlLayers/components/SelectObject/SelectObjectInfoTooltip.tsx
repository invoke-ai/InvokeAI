import { Flex, Icon, ListItem, Text, Tooltip, UnorderedList } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiInfoBold } from 'react-icons/pi';

const Bold = (props: PropsWithChildren) => (
  <Text as="span" fontWeight="semibold">
    {props.children}
  </Text>
);

const SelectObjectHelpTooltipContent = memo(() => {
  const { t } = useTranslation();

  return (
    <Flex gap={3} flexDir="column">
      <Text>
        <Trans i18nKey="controlLayers.selectObject.help1" components={{ Bold: <Bold /> }} />
      </Text>
      <Text>
        <Trans i18nKey="controlLayers.selectObject.help2" components={{ Bold: <Bold /> }} />
      </Text>
      <Text>
        <Trans i18nKey="controlLayers.selectObject.help3" />
      </Text>
      <UnorderedList>
        <ListItem>{t('controlLayers.selectObject.clickToAdd')}</ListItem>
        <ListItem>{t('controlLayers.selectObject.dragToMove')}</ListItem>
        <ListItem>{t('controlLayers.selectObject.clickToRemove')}</ListItem>
      </UnorderedList>
    </Flex>
  );
});

SelectObjectHelpTooltipContent.displayName = 'SelectObjectHelpTooltipContent';

export const SelectObjectInfoTooltip = memo(() => {
  return (
    <Tooltip label={<SelectObjectHelpTooltipContent />}>
      <Flex alignItems="center">
        <Icon as={PiInfoBold} color="base.500" />
      </Flex>
    </Tooltip>
  );
});

SelectObjectInfoTooltip.displayName = 'SelectObjectInfoTooltip';
