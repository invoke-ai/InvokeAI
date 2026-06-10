import { Code, ScrollArea, Stack, Text } from '@chakra-ui/react';
import { Component, type ErrorInfo, type ReactNode } from 'react';

import type { WidgetId } from '../types';
import { Button } from './ui/Button';

interface WidgetFailureBoundaryProps {
  widgetId: WidgetId;
  children: ReactNode;
}

interface WidgetFailureBoundaryState {
  error?: Error;
  details?: string;
}

export class WidgetFailureBoundary extends Component<WidgetFailureBoundaryProps, WidgetFailureBoundaryState> {
  state: WidgetFailureBoundaryState = {};

  static getDerivedStateFromError(error: Error): WidgetFailureBoundaryState {
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
      <Stack bg="bg.surfaceRaised" borderColor="red.500" borderWidth="1px" gap="2" p="3" rounded="md">
        <Text color="red.300" fontSize="xs" fontWeight="700">
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
        <Button
          alignSelf="start"
          size="2xs"
          variant="outline"
          onClick={() => void navigator.clipboard?.writeText(copyableDetails)}
        >
          Copy Error
        </Button>
      </Stack>
    );
  }
}
