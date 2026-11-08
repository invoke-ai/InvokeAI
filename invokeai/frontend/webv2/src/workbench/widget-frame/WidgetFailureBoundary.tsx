import type {
  RegisteredWidget,
  WidgetId,
  WidgetInstanceRuntimeMeta,
  WidgetViewProps,
} from '@workbench/widgetContracts';

import { Box, Code, Flex, HStack, ScrollArea, Stack, Text, useRecipe } from '@chakra-ui/react';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { createLogger } from '@platform/logging/logger';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { toaster } from '@platform/ui/toaster';
import { Tooltip } from '@platform/ui/Tooltip';
import { useScrollAreaPhantomHeal } from '@platform/ui/useScrollAreaPhantomHeal';
import { chipRecipe } from '@theme/recipes';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { isWidgetRegion } from '@workbench/layoutContracts';
import { resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { TriangleAlertIcon } from 'lucide-react';
import {
  Component,
  type ErrorInfo,
  type MouseEvent,
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useRef,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';

import { WidgetPanelFrame, WidgetTooltipFrame } from './WidgetFrames';

const widgetFailureLogger = createLogger({ area: 'widget-render', namespace: 'system' });

interface WidgetFailureBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  isRetrying?: boolean;
  framed?: boolean;
  /** Retain region framing, size, and resize handles on failure; headless hosts omit presentation context. */
  instance?: WidgetInstanceRuntimeMeta;
  presentation?: WidgetViewProps['presentation'];
  /** Attributes the failure to the project hosting the widget. */
  projectId?: string;
  region?: WidgetViewProps['region'];
  resetKey: string;
  widget?: RegisteredWidget;
  widgetId: WidgetId;
  onRetry?: () => void;
  onRequestReload?: () => (() => Promise<void>) | undefined;
  onFailure?: (error: Error) => boolean;
  onRecovered?: () => void;
}

interface WidgetFailureBoundaryState {
  error?: Error;
  details?: string;
  resetKey: string;
}

interface WidgetFailureFallbackProps extends Omit<WidgetFailureBoundaryProps, 'children' | 'resetKey' | 'onRetry'> {
  details: string;
  message: string;
  onCopy: () => Promise<void>;
  onRetry: () => void;
}

const stopRecoveryActivation = (event: KeyboardEvent<HTMLButtonElement>) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.stopPropagation();
  }
};

const WidgetReloadControl = ({
  compact = false,
  isBusy,
  label,
  onRequestReload,
}: {
  compact?: boolean;
  isBusy?: boolean;
  label: string;
  onRequestReload: () => (() => Promise<void>) | undefined;
}) => {
  const { t } = useTranslation();
  const [request, setRequest] = useState<{ run: () => Promise<void> } | null>(null);
  const opener = useRef<HTMLButtonElement | null>(null);
  const close = useCallback(() => setRequest(null), []);
  const confirm = useCallback(async () => {
    await request?.run();
  }, [request]);
  const open = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      const run = onRequestReload();
      if (run) {
        opener.current = event.currentTarget;
        setRequest({ run });
      }
    },
    [onRequestReload]
  );
  const returnFocus = useCallback(() => (opener.current?.getClientRects().length ? opener.current : null), []);
  return (
    <>
      <Button
        aria-label={compact ? t('widgets.failure.compactReloadLabel', { label }) : undefined}
        disabled={isBusy}
        size={compact ? 'xs' : 'sm'}
        variant="outline"
        maxH={compact ? '6' : undefined}
        onClick={open}
        onKeyDown={stopRecoveryActivation}
      >
        {t('widgets.failure.reloadEditor')}
      </Button>
      <ConfirmDialog
        title={t('widgets.failure.reloadTitle')}
        body={t('widgets.failure.reloadWarning')}
        confirmLabel={t('widgets.failure.reloadEditor')}
        isOpen={request !== null}
        onClose={close}
        onConfirm={confirm}
        finalFocusEl={returnFocus}
      />
    </>
  );
};

/** The error card itself: title, expanded detail, and the two recovery verbs. */
const WidgetFailureCard = ({
  details,
  isRetrying,
  label,
  onCopy,
  onRetry,
  onRequestReload,
}: {
  onRequestReload?: WidgetFailureBoundaryProps['onRequestReload'];
  details: string;
  isRetrying?: boolean;
  label: string;
  onCopy: () => Promise<void>;
  onRetry: (event: MouseEvent<HTMLButtonElement>) => void;
}) => {
  const { t } = useTranslation();
  const viewportRef = useRef<HTMLDivElement | null>(null);

  useScrollAreaPhantomHeal(viewportRef);
  const canCopy = canCopyText();
  // The card has no other feedback channel: without a toast the click is
  // indistinguishable from one that silently failed.
  const copy = useCallback(
    () =>
      onCopy().then(
        () => toaster.create({ duration: 2500, title: t('widgets.failure.copiedError'), type: 'success' }),
        () => toaster.create({ title: t('widgets.failure.copyErrorFailed'), type: 'error' })
      ),
    [onCopy, t]
  );

  return (
    <Stack bg="bg.muted" borderColor="border.error" borderWidth="1px" gap="2" p="3" rounded="md">
      <Text color="fg.error" fontSize="md" fontWeight="700">
        {t('widgets.failure.title', { label })}
      </Text>
      <ScrollArea.Root maxH="8rem" variant="hover">
        <ScrollArea.Viewport ref={viewportRef} maxH="8rem">
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
      <Stack direction="row" flexWrap="wrap" gap="2">
        <Button
          aria-busy={isRetrying || undefined}
          loading={isRetrying}
          alignSelf="start"
          size="sm"
          variant="outline"
          onClick={onRetry}
        >
          {t('widgets.failure.retry')}
        </Button>
        <Button
          unavailableReason={canCopy ? undefined : t('common.copyUnavailable')}
          alignSelf="start"
          size="sm"
          variant="outline"
          onClick={copy}
        >
          {t('widgets.failure.copyError')}
        </Button>
        {onRequestReload ? (
          <WidgetReloadControl label={label} isBusy={isRetrying} onRequestReload={onRequestReload} />
        ) : null}
      </Stack>
    </Stack>
  );
};

/** Status-bar scale: the strip is 24px tall, so the detail is available through the keyboard-accessible tooltip. */
export const CompactWidgetFailure = ({
  message,
  isRetrying,
  label,
  onRetry,
  onRequestReload,
}: {
  onRequestReload?: WidgetFailureBoundaryProps['onRequestReload'];
  message: string;
  isRetrying?: boolean;
  label: string;
  onRetry: () => void;
}) => {
  const { t } = useTranslation();
  const recipe = useRecipe({ recipe: chipRecipe });

  const retry = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      const host = event.currentTarget.closest<HTMLElement>(
        '[data-status-bar-widget], [data-floating-window], [data-focus-region]'
      );
      if (host) {
        host.tabIndex = -1;
        host.focus();
      }
      onRetry();
    },
    [onRetry]
  );
  return (
    <Box display="contents">
      {/* Full stacks stay in expanded recovery and diagnostic logs; a compact tooltip must fit the viewport. */}
      <Tooltip content={message}>
        <Button
          aria-label={t('widgets.failure.compactLabel', { label })}
          aria-busy={isRetrying || undefined}
          loading={isRetrying}
          color="fg.error"
          css={recipe()}
          maxH="6"
          size="sm"
          variant="ghost"
          onClick={retry}
          onKeyDown={stopRecoveryActivation}
        >
          <TriangleAlertIcon size={12} />
          <Text data-widget-identity-label="" whiteSpace="nowrap">
            {label}
          </Text>
        </Button>
      </Tooltip>
      {onRequestReload ? (
        <WidgetReloadControl compact label={label} isBusy={isRetrying} onRequestReload={onRequestReload} />
      ) : null}
    </Box>
  );
};

const WidgetFailureHeader = ({ label, region }: { label: string; region: WidgetViewProps['region'] }) => (
  <Box bg={region === 'center' ? 'bg' : 'bg.subtle'} flexShrink="0">
    <HStack borderBottomWidth="1px" h="10" justify="space-between" pe="2" ps="3">
      <HStack color="fg.error" flex="1" gap="1.5" minW="0">
        <TriangleAlertIcon size={14} />
        <Text data-widget-identity-label="" fontSize="md" fontWeight="700">
          {label}
        </Text>
      </HStack>
    </HStack>
  </Box>
);

/** Match loading-frame geometry on failure so status cards do not overflow and panels retain resizing. */
const WidgetFailureFallback = ({
  details,
  message,
  framed,
  isRetrying,
  instance,
  presentation,
  region,
  widget,
  widgetId,
  onCopy,
  onRetry,
  onRequestReload,
}: WidgetFailureFallbackProps) => {
  const { t } = useTranslation();
  const focus = useWorkbenchFocus();
  const retryExpanded = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.currentTarget.closest<HTMLElement>('[data-widget-recovery-host]')?.focus();
      onRetry();
      // Side frames are replaced during loading; the focus owner waits for their recovered frame.
      if (region === 'floating' && instance) {
        focus.focusFloating(instance.id);
      } else if (region && isWidgetRegion(region)) {
        focus.focusRegion(region, instance?.typeId, instance?.id, widget?.implementation.load());
      }
    },
    [focus, instance, onRetry, region, widget]
  );
  const label = widget ? resolveWidgetInstanceLabel(instance ?? {}, widget.manifest, t) : widgetId;

  if (presentation === 'compact') {
    return (
      <CompactWidgetFailure
        isRetrying={isRetrying}
        message={message}
        label={label}
        onRetry={onRetry}
        onRequestReload={onRequestReload}
      />
    );
  }

  if (presentation === 'tooltip' && widget) {
    return (
      <WidgetTooltipFrame icon={widget.manifest.icon}>
        <Text color="fg.error" fontSize="md" fontWeight="700">
          {t('widgets.failure.title', { label })}
        </Text>
        <Text color="fg.muted" fontSize="xs">
          {message}
        </Text>
      </WidgetTooltipFrame>
    );
  }

  const isPanel = region === 'left' || region === 'right' || region === 'bottom';

  if (framed === false || (!isPanel && region !== 'center')) {
    const card = (
      <WidgetFailureCard
        isRetrying={isRetrying}
        details={details}
        label={label}
        onCopy={onCopy}
        onRetry={retryExpanded}
        onRequestReload={onRequestReload}
      />
    );
    return region === 'center' ? (
      <Box p="2" pt="calc(var(--wb-center-chrome-inset, 0px) + var(--chakra-spacing-2))">
        {card}
      </Box>
    ) : (
      card
    );
  }

  const frameContent = (
    <>
      {isPanel && widget?.manifest.chrome?.header !== 'hidden' ? (
        <WidgetFailureHeader label={label} region={region} />
      ) : null}
      <Box
        flex="1"
        minH="0"
        overflow="auto"
        p="2"
        pt={region === 'center' ? 'calc(var(--wb-center-chrome-inset, 0px) + var(--chakra-spacing-2))' : undefined}
      >
        <WidgetFailureCard
          isRetrying={isRetrying}
          details={details}
          label={label}
          onCopy={onCopy}
          onRetry={retryExpanded}
          onRequestReload={onRequestReload}
        />
      </Box>
    </>
  );

  return isPanel ? (
    <WidgetPanelFrame
      instanceId={instance?.id}
      label={t('widgets.failure.title', { label })}
      region={region}
      typeId={instance?.typeId}
    >
      {frameContent}
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
      {frameContent}
    </Flex>
  );
};

export class WidgetFailureBoundary extends Component<WidgetFailureBoundaryProps, WidgetFailureBoundaryState> {
  state: WidgetFailureBoundaryState = { resetKey: this.props.resetKey };

  private handleRetry = () => {
    this.props.onRetry?.();
    this.setState({ details: undefined, error: undefined, resetKey: this.props.resetKey });
  };

  private handleCopyError = (): Promise<void> => {
    const { details, error } = this.state;

    return error ? copyText(details ?? error.message) : Promise.reject(new Error('No error to copy.'));
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
    const shouldLog = this.props.onFailure?.(error) ?? true;
    if (shouldLog) {
      widgetFailureLogger.child({ projectId: this.props.projectId }).error({
        context: {
          componentStack: errorInfo.componentStack,
          instanceId: this.props.instance?.id,
          region: this.props.region,
          widgetId: this.props.widgetId,
        },
        error,
        message: `Widget ${this.props.widgetId} failed to render`,
        name: 'widget.render-failed',
      });
    }
    this.setState({ details: [error.stack ?? error.message, errorInfo.componentStack].filter(Boolean).join('\n') });
  }

  componentDidMount() {
    if (!this.state.error) {
      this.props.onRecovered?.();
    }
  }

  componentDidUpdate() {
    if (!this.state.error) {
      this.props.onRecovered?.();
    }
  }

  render() {
    const { children, framed, isRetrying, instance, presentation, region, widget, widgetId, onRequestReload } =
      this.props;
    const { details, error } = this.state;

    if (!error) {
      return children;
    }
    if (this.props.fallback !== undefined) {
      return this.props.fallback;
    }

    return (
      <Box data-testid="widget-failure" display="contents">
        <WidgetFailureFallback
          details={details ?? error.message}
          message={error.message}
          framed={framed}
          isRetrying={isRetrying}
          instance={instance}
          presentation={presentation}
          region={region}
          widget={widget}
          widgetId={widgetId}
          onCopy={this.handleCopyError}
          onRetry={this.handleRetry}
          onRequestReload={onRequestReload}
        />
      </Box>
    );
  }
}
