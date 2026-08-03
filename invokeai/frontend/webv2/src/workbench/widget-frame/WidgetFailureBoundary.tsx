import type {
  RegisteredWidget,
  WidgetId,
  WidgetInstanceRuntimeMeta,
  WidgetViewProps,
} from '@workbench/widgetContracts';

import { Box, Code, Flex, HStack, ScrollArea, Stack, Text, useRecipe } from '@chakra-ui/react';
import { Button } from '@platform/ui';
import { chipRecipe } from '@theme/recipes';
import { resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { TriangleAlertIcon } from 'lucide-react';
import { Component, type ErrorInfo, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { WidgetPanelFrame, WidgetTooltipFrame } from './WidgetFrames';

interface WidgetFailureBoundaryProps {
  children: ReactNode;
  /**
   * Presentation context, mirroring what `WidgetLoadingFallback` receives. When
   * supplied, the fallback keeps the widget's own frame — panel width, overflow
   * clamp and resize handle survive the crash. Headless hosts omit it.
   */
  instance?: WidgetInstanceRuntimeMeta;
  presentation?: WidgetViewProps['presentation'];
  region?: WidgetViewProps['region'];
  resetKey: string;
  widget?: RegisteredWidget;
  widgetId: WidgetId;
  onRetry?: () => void;
}

interface WidgetFailureBoundaryState {
  error?: Error;
  details?: string;
  resetKey: string;
}

interface WidgetFailureFallbackProps extends Omit<WidgetFailureBoundaryProps, 'children' | 'resetKey' | 'onRetry'> {
  details: string;
  onCopy: () => void;
  onRetry: () => void;
}

/** The error card itself: title, expanded detail, and the two recovery verbs. */
const WidgetFailureCard = ({
  details,
  label,
  onCopy,
  onRetry,
}: {
  details: string;
  label: string;
  onCopy: () => void;
  onRetry: () => void;
}) => {
  const { t } = useTranslation();

  return (
    <Stack bg="bg.muted" borderColor="border.error" borderWidth="1px" gap="2" p="3" rounded="md">
      <Text color="fg.error" fontSize="xs" fontWeight="700">
        {t('widgets.failure.title', { label })}
      </Text>
      <ScrollArea.Root maxH="8rem" size="xs" variant="hover">
        <ScrollArea.Viewport maxH="8rem">
          <ScrollArea.Content>
            <Code display="block" p="2" whiteSpace="pre-wrap">
              {details}
            </Code>
          </ScrollArea.Content>
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar>
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
        <ScrollArea.Scrollbar orientation="horizontal">
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
        <ScrollArea.Corner />
      </ScrollArea.Root>
      <Stack direction="row" gap="2">
        <Button alignSelf="start" size="2xs" variant="outline" onClick={onRetry}>
          {t('widgets.failure.retry')}
        </Button>
        <Button alignSelf="start" size="2xs" variant="outline" onClick={onCopy}>
          {t('widgets.failure.copyError')}
        </Button>
      </Stack>
    </Stack>
  );
};

/** Status-bar scale: the strip is 24px tall, so the detail rides in the title tooltip. */
const CompactWidgetFailure = ({ details, label, onRetry }: { details: string; label: string; onRetry: () => void }) => {
  const { t } = useTranslation();
  const recipe = useRecipe({ recipe: chipRecipe });

  return (
    <HStack
      aria-label={t('widgets.failure.compactLabel', { label })}
      color="fg.error"
      css={recipe()}
      role="status"
      title={details}
      onClick={onRetry}
    >
      <TriangleAlertIcon size={12} />
      <Text data-widget-identity-label="" whiteSpace="nowrap">
        {label}
      </Text>
    </HStack>
  );
};

const WidgetFailureHeader = ({ label, region }: { label: string; region: WidgetViewProps['region'] }) => (
  <Box bg={region === 'center' ? 'bg' : 'bg.subtle'} flexShrink="0">
    <HStack borderBottomWidth="1px" h="10" justify="space-between" pe="2" ps="3">
      <HStack color="fg.error" flex="1" gap="1.5" minW="0">
        <TriangleAlertIcon size={14} />
        <Text data-widget-identity-label="" fontSize="xs" fontWeight="700">
          {label}
        </Text>
      </HStack>
    </HStack>
  </Box>
);

/**
 * Presentation- and region-aware failure UI, the twin of `WidgetLoadingFallback`.
 * Rendering the bare card everywhere used to blow a crashed widget out of its
 * region: in the status bar it overflowed a 24px strip, and in a side panel it
 * destroyed the frame that carries the panel width and the resize handle.
 */
const WidgetFailureFallback = ({
  details,
  instance,
  presentation,
  region,
  widget,
  widgetId,
  onCopy,
  onRetry,
}: WidgetFailureFallbackProps) => {
  const { t } = useTranslation();
  const label = widget ? resolveWidgetInstanceLabel(instance ?? {}, widget.manifest, t) : widgetId;

  if (presentation === 'compact') {
    return <CompactWidgetFailure details={details} label={label} onRetry={onRetry} />;
  }

  if (presentation === 'tooltip' && widget) {
    return (
      <WidgetTooltipFrame icon={widget.manifest.icon}>
        <Text color="fg.error" fontSize="xs" fontWeight="700">
          {t('widgets.failure.title', { label })}
        </Text>
        <Text color="fg.muted" fontSize="2xs">
          {details}
        </Text>
      </WidgetTooltipFrame>
    );
  }

  const isPanel = region === 'left' || region === 'right' || region === 'bottom';

  if (!isPanel && region !== 'center') {
    return <WidgetFailureCard details={details} label={label} onCopy={onCopy} onRetry={onRetry} />;
  }

  const framed = (
    <>
      {widget?.manifest.chrome?.header === 'hidden' ? null : <WidgetFailureHeader label={label} region={region} />}
      <Box flex="1" minH="0" overflow="auto" p="2">
        <WidgetFailureCard details={details} label={label} onCopy={onCopy} onRetry={onRetry} />
      </Box>
    </>
  );

  // Keeping the widget's own frame is the whole point: the panel frame carries
  // the region width, the overflow clamp and the resize handle.
  return isPanel ? (
    <WidgetPanelFrame instanceId={instance?.id} region={region} typeId={instance?.typeId}>
      {framed}
    </WidgetPanelFrame>
  ) : (
    <Flex
      bg="bg.inset"
      data-hotkey-widget-instance-id={instance?.id}
      data-hotkey-widget-region={region}
      data-hotkey-widget-type-id={instance?.typeId}
      direction="column"
      h="full"
      minH="0"
      w="full"
    >
      {framed}
    </Flex>
  );
};

export class WidgetFailureBoundary extends Component<WidgetFailureBoundaryProps, WidgetFailureBoundaryState> {
  state: WidgetFailureBoundaryState = { resetKey: this.props.resetKey };

  private handleRetry = () => {
    this.props.onRetry?.();
    this.setState({ details: undefined, error: undefined, resetKey: this.props.resetKey });
  };

  private handleCopyError = () => {
    const { details, error } = this.state;

    if (error) {
      void navigator.clipboard?.writeText(details ?? error.message);
    }
  };

  static getDerivedStateFromProps(
    props: WidgetFailureBoundaryProps,
    state: WidgetFailureBoundaryState
  ): Partial<WidgetFailureBoundaryState> | null {
    if (props.resetKey !== state.resetKey) {
      return { details: undefined, error: undefined, resetKey: props.resetKey };
    }

    return null;
  }

  static getDerivedStateFromError(error: Error): Partial<WidgetFailureBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ details: errorInfo.componentStack ?? error.stack ?? error.message });
  }

  render() {
    const { children, instance, presentation, region, widget, widgetId } = this.props;
    const { details, error } = this.state;

    if (!error) {
      return children;
    }

    return (
      <Box data-testid="widget-failure" display="contents">
        <WidgetFailureFallback
          details={details ?? error.message}
          instance={instance}
          presentation={presentation}
          region={region}
          widget={widget}
          widgetId={widgetId}
          onCopy={this.handleCopyError}
          onRetry={this.handleRetry}
        />
      </Box>
    );
  }
}
