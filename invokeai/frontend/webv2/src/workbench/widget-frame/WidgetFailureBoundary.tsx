import type { WidgetId } from '@workbench/widgetContracts';

import { Code, ScrollArea, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui';
import { Component, type ErrorInfo, type ReactNode } from 'react';

interface WidgetFailureBoundaryProps {
  widgetId: WidgetId;
  resetKey: string;
  children: ReactNode;
  onRetry?: () => void;
}

interface WidgetFailureBoundaryState {
  error?: Error;
  details?: string;
  resetKey: string;
}

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
    const { children, widgetId } = this.props;
    const { details, error } = this.state;

    if (!error) {
      return children;
    }

    const copyableDetails = details ?? error.message;

    return (
      <Stack bg="bg.muted" borderColor="border.error" borderWidth="1px" gap="2" p="3" rounded="md">
        <Text color="fg.error" fontSize="xs" fontWeight="700">
          Widget failed: {widgetId}
        </Text>
        <ScrollArea.Root maxH="8rem" size="xs" variant="hover">
          <ScrollArea.Viewport maxH="8rem">
            <ScrollArea.Content>
              <Code display="block" p="2" whiteSpace="pre-wrap">
                {copyableDetails}
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
          <Button alignSelf="start" size="2xs" variant="outline" onClick={this.handleRetry}>
            Retry
          </Button>
          <Button alignSelf="start" size="2xs" variant="outline" onClick={this.handleCopyError}>
            Copy Error
          </Button>
        </Stack>
      </Stack>
    );
  }
}
