import type { RegisteredWidget, WidgetInstanceRuntimeMeta, WidgetViewProps } from '@workbench/widgetContracts';

import { Box, Flex, HStack, Text, useRecipe } from '@chakra-ui/react';
import { chipRecipe } from '@theme/recipes';
import { resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { useTranslation } from 'react-i18next';

import { WidgetPanelFrame, WidgetTooltipFrame } from './WidgetFrames';
import { WidgetIdentityIcon } from './WidgetIdentityIcon';

interface WidgetLoadingFallbackProps {
  instance: WidgetInstanceRuntimeMeta;
  presentation?: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  widget: RegisteredWidget;
}

const WidgetLoadingHeader = ({
  label,
  region,
  widget,
}: {
  label: string;
  region: WidgetViewProps['region'];
  widget: RegisteredWidget;
}) => (
  <Box bg={region === 'center' ? 'bg' : 'bg.subtle'} flexShrink="0">
    <HStack borderBottomWidth="1px" h="10" justify="space-between" pe="2" ps="3">
      <HStack flex="1" gap="1.5" minW="0">
        <WidgetIdentityIcon icon={widget.manifest.icon} isLoading />
        <Text data-widget-identity-label="" fontSize="xs" fontWeight="700">
          {label}
        </Text>
      </HStack>
    </HStack>
  </Box>
);

const WidgetLoadingSurface = ({ bg = 'bg.inset', label }: { bg?: 'bg.inset' | 'bg.subtle'; label: string }) => (
  <Box
    aria-atomic="true"
    aria-busy="true"
    aria-label={label}
    aria-live="polite"
    bg={bg}
    flex="1"
    minH="0"
    minW="0"
    role="status"
    w="full"
  />
);

const CompactWidgetLoadingFallback = ({
  label,
  loadingLabel,
  widget,
}: {
  label: string;
  loadingLabel: string;
  widget: RegisteredWidget;
}) => {
  const recipe = useRecipe({ recipe: chipRecipe });

  return (
    <HStack
      aria-atomic="true"
      aria-busy="true"
      aria-label={loadingLabel}
      aria-live="polite"
      color="fg.subtle"
      css={recipe()}
      opacity="0.65"
      role="status"
    >
      <WidgetIdentityIcon icon={widget.manifest.icon} isLoading />
      <Text data-widget-identity-label="" whiteSpace="nowrap">
        {label}
      </Text>
    </HStack>
  );
};

const TooltipWidgetLoadingFallback = ({
  label,
  loadingLabel,
  widget,
}: {
  label: string;
  loadingLabel: string;
  widget: RegisteredWidget;
}) => (
  <Box aria-atomic="true" aria-busy="true" aria-label={loadingLabel} aria-live="polite" role="status">
    <WidgetTooltipFrame icon={widget.manifest.icon} isLoading>
      <Text data-widget-identity-label="" fontSize="xs" fontWeight="700">
        {label}
      </Text>
    </WidgetTooltipFrame>
  </Box>
);

const InlineWidgetLoadingFallback = ({
  label,
  loadingLabel,
  widget,
}: {
  label: string;
  loadingLabel: string;
  widget: RegisteredWidget;
}) => (
  <Flex
    align="center"
    aria-atomic="true"
    aria-busy="true"
    aria-label={loadingLabel}
    aria-live="polite"
    color="fg.subtle"
    gap="1.5"
    h="full"
    justify="center"
    minH="8rem"
    role="status"
    w="full"
  >
    <WidgetIdentityIcon icon={widget.manifest.icon} isLoading />
    <Text data-widget-identity-label="" fontSize="xs" fontWeight="700">
      {label}
    </Text>
  </Flex>
);

export const WidgetLoadingFallback = ({ instance, presentation, region, widget }: WidgetLoadingFallbackProps) => {
  const { t } = useTranslation();
  const label = resolveWidgetInstanceLabel(instance, widget.manifest, t);
  const loadingLabel = t('widgets.loadingLabel', { label });

  if (presentation === 'compact') {
    return <CompactWidgetLoadingFallback label={label} loadingLabel={loadingLabel} widget={widget} />;
  }

  if (presentation === 'tooltip') {
    return <TooltipWidgetLoadingFallback label={label} loadingLabel={loadingLabel} widget={widget} />;
  }

  if (region === 'left' || region === 'right' || region === 'bottom') {
    return (
      <WidgetPanelFrame instanceId={instance.id} region={region} typeId={instance.typeId}>
        {widget.manifest.chrome?.header === 'hidden' ? null : (
          <WidgetLoadingHeader label={label} region={region} widget={widget} />
        )}
        <WidgetLoadingSurface bg="bg.subtle" label={loadingLabel} />
      </WidgetPanelFrame>
    );
  }

  if (region === 'center') {
    return (
      <Flex
        bg="bg.inset"
        data-hotkey-widget-instance-id={instance.id}
        data-hotkey-widget-region={region}
        data-hotkey-widget-type-id={instance.typeId}
        direction="column"
        h="full"
        minH="0"
        w="full"
      >
        {widget.manifest.chrome?.header === 'hidden' ? null : (
          <WidgetLoadingHeader label={label} region={region} widget={widget} />
        )}
        <WidgetLoadingSurface label={loadingLabel} />
      </Flex>
    );
  }

  return <InlineWidgetLoadingFallback label={label} loadingLabel={loadingLabel} widget={widget} />;
};
